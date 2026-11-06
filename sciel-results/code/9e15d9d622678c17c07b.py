"""Strategy: Robust Polynomial Fit using Legendre Basis with IRLS Outlier Down-weighting."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 10:
        return np.zeros(6)
    
    deg = 5
    
    # 1. Scale x to [-1, 1] for Legendre polynomial stability
    # This avoids the ill-conditioning of monomial Vandermonde matrices for high degrees
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case: constant x. Fit a constant.
        c0 = np.mean(y)
        coeffs = np.zeros(6)
        coeffs[0] = c0
        return coeffs

    # Map x to t in [-1, 1]
    t = 2.0 * (x - x_min) / x_range - 1.0
    
    # 2. Construct Legendre design matrix
    # P_k(t) for k=0..5
    # eval_legendre returns the value of the k-th Legendre polynomial at t
    # We need a matrix where row i is [P_0(t_i), P_1(t_i), ..., P_5(t_i)]
    L = np.zeros((n, deg + 1))
    for k in range(deg + 1):
        L[:, k] = eval_legendre(k, t)
    
    # 3. Initial Least Squares Fit to get residuals for weighting
    try:
        coeffs_legendre, residuals, rank, sv = np.linalg.lstsq(L, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(6)
    
    # 4. Iterative Reweighted Least Squares (IRLS) for Robustness
    # Use Tukey's biweight or Huber loss? 
    # Simple approach: iterative reweighting based on residuals.
    # Weights w_i = 1 / (1 + (r_i / sigma)^2) or similar.
    # Let's use a robust scale estimate (MAD) to set the threshold.
    
    for iteration in range(5):
        if time.time() > deadline - 5.0:
            break
            
        y_pred = L @ coeffs_legendre
        residuals_vec = y - y_pred
        
        # Estimate scale using MAD (Median Absolute Deviation)
        med_res = np.median(residuals_vec)
        mad_res = np.median(np.abs(residuals_vec - med_res))
        
        if mad_res < 1e-12:
            # If MAD is very small, fall back to std or just stop
            std_res = np.std(residuals_vec)
            if std_res < 1e-12:
                break
            scale = std_res
        else:
            # Convert MAD to standard deviation equivalent for normal dist
            # sigma ~ 1.4826 * MAD
            scale = 1.4826 * mad_res
        
        # Compute weights
        # Use a soft thresholding weight function
        # w = 1 if |r| < 3*scale, else decays
        # Simple robust weight: w = 1 / (1 + (r/scale)^2) is Cauchy-like
        # Or Huber: w = 1 if |r| < k*scale, else k*scale/|r|
        # Let's use a simple inverse square weight which is smooth and robust
        # w_i = 1 / (1 + (r_i / (2*scale))^2)
        
        r_scaled = residuals_vec / (2.0 * scale)
        weights = 1.0 / (1.0 + r_scaled**2)
        
        # Ensure weights are not zero to avoid division issues in weighted lstsq
        weights = np.maximum(weights, 1e-6)
        
        # Weighted Least Squares
        # Minimize sum(w_i * (y_i - L_i @ c)^2)
        # This is equivalent to solving (L^T W L) c = L^T W y
        # Using sqrt(weights) to transform to standard LS:
        # (sqrt(W) L) c = sqrt(W) y
        
        sqrt_w = np.sqrt(weights)
        L_w = L * sqrt_w[:, np.newaxis]
        y_w = y * sqrt_w
        
        try:
            coeffs_legendre_new, _, _, _ = np.linalg.lstsq(L_w, y_w, rcond=None)
        except np.linalg.LinAlgError:
            break
            
        # Check convergence
        if np.linalg.norm(coeffs_legendre_new - coeffs_legendre) < 1e-8:
            coeffs_legendre = coeffs_legendre_new
            break
        else:
            coeffs_legendre = coeffs_legendre_new
            
    # 5. Convert Legendre coefficients to Monomial coefficients
    # We have p(t) = sum_{k=0}^5 c_k P_k(t)
    # We need q(x) = sum_{i=0}^5 a_i x^i such that q(x) = p(t(x))
    # t = 2(x-x_min)/x_range - 1 = (2/x_range)x - (2x_min/x_range + 1)
    # Let A = 2/x_range, B = -(2x_min/x_range + 1)
    # t = A*x + B
    
    A = 2.0 / x_range
    B = -(2.0 * x_min / x_range + 1.0)
    
    # We can evaluate the polynomial at specific points to recover monomial coeffs
    # Or use the expansion of Legendre polynomials in powers of x.
    # P_k(t) is a polynomial in t of degree k.
    # P_k(A*x+B) is a polynomial in x of degree k.
    # We can build the transformation matrix T such that:
    # [P_0(t), P_1(t), ..., P_5(t)]^T = T * [1, x, x^2, x^3, x^4, x^5]^T ? No.
    # Actually: P_k(t) = sum_{j=0}^k M_{k,j} x^j
    # Then p(t) = sum_k c_k sum_j M_{k,j} x^j = sum_j (sum_k c_k M_{k,j}) x^j
    # So a_j = sum_k c_k M_{k,j}
    
    # Let's compute M_{k,j} by expanding P_k(A*x+B)
    # We can do this by evaluating P_k at points and fitting, or analytically.
    # Since deg is small (5), we can just evaluate the resulting polynomial
    # at 6 distinct points in the domain and solve for monomial coeffs.
    
    # Choose 6 points in x domain, e.g., equally spaced
    x_eval = np.linspace(x_min, x_max, 6)
    t_eval = 2.0 * (x_eval - x_min) / x_range - 1.0
    
    # Evaluate Legendre polynomials at these points
    L_eval = np.zeros((6, deg + 1))
    for k in range(deg + 1):
        L_eval[:, k] = eval_legendre(k, t_eval)
    
    # Values of the fitted polynomial at these points
    p_values = L_eval @ coeffs_legendre
    
    # Now fit monomial coeffs a_i to match p_values at x_eval
    # V_eval * a = p_values
    V_eval = np.vander(x_eval, N=deg + 1, increasing=True)
    
    try:
        coeffs_monomial, _, _, _ = np.linalg.lstsq(V_eval, p_values, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback: if conversion fails, return zeros or try direct fit?
        # Direct fit might be unstable, but better than zeros.
        # Let's try direct fit on original x if this fails.
        x_scale = np.max(np.abs(x))
        if x_scale < 1e-12: x_scale = 1.0
        x_scaled = x / x_scale
        V_orig = np.vander(x_scaled, N=deg + 1, increasing=True)
        try:
            c_scaled, _, _, _ = np.linalg.lstsq(V_orig, y, rcond=None)
            coeffs_monomial = c_scaled * (x_scale ** (-np.arange(deg + 1)))
        except:
            return np.zeros(6)

    # 6. Final check and return
    if time.time() > deadline:
        return coeffs_monomial
        
    return coeffs_monomial
# EVOLVE-BLOCK-END
