"""Rewritten program: Tikhonov-regularized degree-5 polynomial fit with internal k-fold CV and standardization."""
import numpy as np


def _build_design_matrix(x, degree):
    """Build Vandermonde matrix in ascending powers."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    # Use np.vander with increasing=True for ascending powers
    V = np.vander(x, N=degree + 1, increasing=True)
    return V


def _ridge_solve(V, y, lam, penalize_intercept=False):
    """
    Solve ridge regression: (V^T V + lam * R) w = V^T y
    R is a diagonal matrix. If penalize_intercept is False, R[0,0]=0, else R[0,0]=1.
    """
    n_params = V.shape[1]
    A = V.T @ V
    b = V.T @ y
    
    # Construct regularization matrix R
    # By default, we do NOT penalize the intercept (index 0)
    if penalize_intercept:
        R = np.eye(n_params)
    else:
        R = np.eye(n_params)
        R[0, 0] = 0.0
        
    A_reg = A + lam * R
    
    # Ensure A_reg is well-conditioned
    try:
        w = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        w = np.linalg.lstsq(A_reg, b, rcond=None)[0]
        
    return w

def _transform_coeffs_scaled_to_original(w_scaled, mu, sigma, degree):
    """
    Transform coefficients from standardized basis z = (x-mu)/sigma 
    to original basis x.
    y = sum_{i=0}^D w_scaled[i] * z^i
    z^i = ( (x-mu)/sigma )^i = sigma^-i * sum_{j=0}^i binom(i,j) x^j (-mu)^(i-j)
    """
    w_orig = np.zeros(degree + 1)
    for i in range(degree + 1):
        if w_scaled[i] == 0:
            continue
        coeff_i = w_scaled[i] * (sigma ** (-i))
        # Expand (x - mu)^i
        # Term j in x^j comes from binom(i, j) * (-mu)^(i-j)
        for j in range(i + 1):
            binom_val = 1.0
            # Calculate binom(i, j)
            # For small i (max 5), we can compute directly
            if j == 0 or j == i:
                binom_val = 1.0
            elif j == 1 or j == i - 1:
                binom_val = float(i)
            elif i == 4 and j == 2:
                binom_val = 6.0
            elif i == 5 and j == 2:
                binom_val = 10.0
            elif i == 5 and j == 3:
                binom_val = 10.0
            else:
                # General case for safety, though i<=5
                import math
                binom_val = float(math.comb(i, j))
            
            term = binom_val * ((-mu) ** (i - j))
            w_orig[j] += coeff_i * term
    return w_orig


def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(training_x)
    degree = 5
    
    # Handle edge cases
    if n < degree + 1:
        if n == 0:
            return np.zeros(6)
        deg_eff = min(degree, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(training_x, training_subgrid_tendency, deg_eff)
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except:
            return np.zeros(6)

    # 1. Standardize x for better conditioning and meaningful regularization
    mu = np.mean(training_x)
    sigma = np.std(training_x)
    if sigma < 1e-10:
        sigma = 1.0
        
    x_scaled = (training_x - mu) / sigma
    
    # 2. Build design matrix for scaled x
    V = _build_design_matrix(x_scaled, degree)
    
    # 3. Determine lambda range
    # With standardized x, the scale of V^T V is roughly n * (1 + small terms)
    # Eigenvalues are O(n).
    # Lambda should be in a range that balances bias/variance.
    # Typical ridge lambdas for standardized data: 1e-6 to 1e6?
    # Let's use a wide logspace.
    lambdas = np.logspace(-8, 8, 50)
    
    # 4. K-fold cross-validation
    k = 10
    if n < k:
        k = max(2, n // 2)
        if k < 2:
            lam = 1e-3
            w_scaled_cv = _ridge_solve(V, training_subgrid_tendency, lam, penalize_intercept=False)
            w_orig = _transform_coeffs_scaled_to_original(w_scaled_cv, mu, sigma, degree)
            return w_orig
    
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    fold_indices = np.array_split(indices, k)
    
    best_mse = np.inf
    best_w_scaled = np.zeros(6)
    best_lam = 0.0
    
    for lam in lambdas:
        mse_sum = 0.0
        for i in range(k):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            
            V_train = V[train_idx]
            y_train = training_subgrid_tendency[train_idx]
            V_val = V[val_idx]
            y_val = training_subgrid_tendency[val_idx]
            
            w_cv = _ridge_solve(V_train, y_train, lam, penalize_intercept=False)
            pred_val = V_val @ w_cv
            mse = np.mean((pred_val - y_val) ** 2)
            mse_sum += mse
        
        avg_mse = mse_sum / k
        
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_lam = lam
            best_w_scaled = _ridge_solve(V, training_subgrid_tendency, best_lam, penalize_intercept=False)
            
    # 5. Transform best coefficients back to original x basis
    w_orig = _transform_coeffs_scaled_to_original(best_w_scaled, mu, sigma, degree)
    
    return w_orig
