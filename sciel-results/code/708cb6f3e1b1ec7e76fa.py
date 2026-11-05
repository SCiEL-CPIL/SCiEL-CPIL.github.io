"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import stats

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model using Quantile Regression (Median)
    on a Chebyshev basis for numerical stability.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        return np.zeros(6)
    
    # Determine the range of x to map to [-1, 1] for Chebyshev polynomials
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Avoid division by zero if range is tiny
    if x_max - x_min < 1e-12:
        x_center = x_min
        x_scale = 1.0
    else:
        x_center = (x_max + x_min) / 2.0
        x_scale = (x_max - x_min) / 2.0
    
    # Map x to t in [-1, 1]
    t = (x - x_center) / x_scale
    
    # Generate Chebyshev design matrix T(t) for degree 0 to 5
    def chebyshev_matrix(t_vals, degree):
        n = len(t_vals)
        T = np.zeros((n, degree + 1))
        T[:, 0] = 1.0
        if degree >= 1:
            T[:, 1] = t_vals
        for k in range(2, degree + 1):
            T[:, k] = 2.0 * t_vals * T[:, k-1] - T[:, k-2]
        return T
    
    degree = 5
    T = chebyshev_matrix(t, degree)
    
    # Check time before expensive quantile regression
    if time.time() > DEADLINE - 20:
        # Fallback to simple mean if time is tight
        return np.array([np.mean(y)] + [0.0]*5)
    
    # Solve Quantile Regression for median (tau=0.5)
    # This minimizes sum |y_i - T_i @ beta|
    # scipy.stats.mstats.linregress is for simple linear, we need general quantile regression
    # We can use scipy.optimize or linear programming, but for small dimensions (6 coeffs)
    # and moderate N, a linear programming approach or specialized solver is best.
    # However, scipy doesn't have a direct 'quantile_regression' for multivariate.
    # We can use scipy.optimize.linprog to solve min sum(u+v) s.t. T@b - u + v = y, u,v>=0
    # But N can be large. Let's check size.
    
    N = len(x)
    
    # If N is very large, LP might be slow. 
    # Alternative: Use scipy.stats.mstats.ols_linear? No.
    # Let's try a simpler robust approach first: Iterative Reweighted Least Squares (IRLS) for L1.
    # Or just use scipy.optimize.minimize with SLSQP or Nelder-Mead on L1 loss?
    # Given the deadline, let's try a robust OLS variant: Huber loss or simply Median via LP if N < 50000.
    
    # For safety and speed, let's implement a simple IRLS for L1 (Quantile Regression)
    # Initial guess: OLS
    try:
        beta_ols, residuals, rank, sv = np.linalg.lstsq(T, y, rcond=None)
    except Exception:
        beta_ols = np.zeros(degree + 1)
    
    # IRLS for L1 (Median regression)
    # Weights w_i = 1 / max(|r_i|, eps)
    # Iterate: beta_new = (T^T W T)^-1 T^T W y
    
    beta = beta_ols.copy()
    max_iter = 50
    tol = 1e-6
    eps = 1e-8
    
    for i in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        # Compute residuals
        r = y - T.dot(beta)
        
        # Compute weights
        # For L1, w = 1/|r|
        # Clip to avoid division by zero and extreme weights
        abs_r = np.abs(r)
        abs_r = np.maximum(abs_r, eps)
        w = 1.0 / abs_r
        
        # Weighted Least Squares
        # Minimize sum w_i (y_i - T_i beta)^2
        # This approximates L1 if w is updated correctly.
        
        # Construct W
        # TWT = T^T (diag(w)) T
        # Tw = T^T (w * y)
        
        # Efficient computation:
        # TWT = T.T * (w * T)  -> (6, N) * (N,) * (N, 6) -> (6, N) * (N, 6)
        # Actually: T.T @ (w[:, None] * T)
        
        try:
            TWT = T.T @ (w[:, None] * T)
            Tw = T.T @ (w * y)
            
            # Regularize slightly to ensure invertibility
            TWT += 1e-8 * np.eye(degree + 1)
            
            beta_new = np.linalg.solve(TWT, Tw)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(TWT, Tw, rcond=None)[0]
        
        # Check convergence
        if np.linalg.norm(beta_new - beta) < tol * (1 + np.linalg.norm(beta)):
            beta = beta_new
            break
        
        beta = beta_new
    
    cheb_coeffs = beta
    
    # Convert Chebyshev coefficients to standard monomial coefficients
    alpha = 1.0 / x_scale
    beta_const = -x_center / x_scale  # Renamed to avoid conflict with coefficient beta
    
    # We represent each T_k(alpha*x + beta_const) as a vector of monomial coeffs [c_0, c_1, ..., c_5]
    monomial_coeffs = np.zeros((degree + 1, degree + 1))
    
    # T_0(alpha*x + beta_const) = 1
    monomial_coeffs[0, 0] = 1.0
    
    # T_1(alpha*x + beta_const) = alpha*x + beta_const
    monomial_coeffs[1, 0] = beta_const
    monomial_coeffs[1, 1] = alpha
    
    for k in range(2, degree + 1):
        # T_k(z) = 2*z*T_{k-1}(z) - T_{k-2}(z)
        # z = alpha*x + beta_const
        # 2*z*T_{k-1}(z) = 2*(alpha*x + beta_const) * sum_{j=0}^{k-1} monomial_coeffs[k-1, j] x^j
        #                 = sum_{j=0}^{k-1} 2*alpha*monomial_coeffs[k-1, j] x^{j+1} 
        #                   + sum_{j=0}^{k-1} 2*beta_const*monomial_coeffs[k-1, j] x^j
        
        temp = np.zeros(degree + 1)
        
        # Contribution from 2*beta_const * T_{k-1}(z)
        for j in range(k):
            temp[j] += 2.0 * beta_const * monomial_coeffs[k-1, j]
        
        # Contribution from 2*alpha * x * T_{k-1}(z)
        for j in range(k):
            temp[j+1] += 2.0 * alpha * monomial_coeffs[k-1, j]
        
        # Subtract T_{k-2}(z)
        for j in range(k-1):
            temp[j] -= monomial_coeffs[k-2, j]
        
        monomial_coeffs[k] = temp
    
    # Now, the total monomial coefficients are:
    # c_total = sum_{k=0}^D a_k * monomial_coeffs[k]
    final_coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        final_coeffs += cheb_coeffs[k] * monomial_coeffs[k]
    
    # Check for numerical stability
    if not np.all(np.isfinite(final_coeffs)):
        # If conversion failed, fall back to OLS monomial coefficients
        try:
            # Convert OLS cheb coeffs to monomial
            final_coeffs_ols = np.zeros(degree + 1)
            for k in range(degree + 1):
                final_coeffs_ols += beta_ols[k] * monomial_coeffs[k]
            
            if np.all(np.isfinite(final_coeffs_ols)):
                final_coeffs = final_coeffs_ols
            else:
                final_coeffs = np.array([np.mean(y)] + [0.0]*5)
        except Exception:
            final_coeffs = np.array([np.mean(y)] + [0.0]*5)
            
    return final_coeffs
# EVOLVE-BLOCK-END
