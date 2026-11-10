"""Rewritten program: physics-informed sparse basis with L1 selection for unresolved tendency closure."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    
    Uses a physics-informed approach: constructs a candidate basis including
    polynomial terms up to degree 5, applies L1 regularization (Lasso) to
    select the most relevant terms, then refits with least squares on the
    selected subset. The final coefficients are returned in ascending powers.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten and validate
    x = x.ravel()
    y = y.ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Build full polynomial design matrix (degrees 0-5)
    V = np.vander(x, N=6, increasing=True)  # columns: x^0, x^1, ..., x^5
    
    # Normalize columns for stable L1/Lasso fitting
    col_means = V.mean(axis=0)
    col_stds = V.std(axis=0)
    col_stds[col_stds < 1e-12] = 1.0
    V_norm = (V - col_means) / col_stds
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_norm = np.zeros_like(y)
    else:
        y_norm = (y - y_mean) / y_std
    
    # Try multiple L1 regularization strengths and pick the one that best
    # balances sparsity and fit quality. We use a simple approach:
    # try several alpha values, compute training MSE, and pick the alpha
    # that gives good fit with reasonable sparsity.
    
    best_coeff = np.zeros(6)
    best_score = np.inf
    
    # A range of regularization strengths (in normalized space)
    # We want to encourage sparsity but not over-regularize
    alphas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    for alpha in alphas:
        # Simple Lasso via coordinate descent (soft-thresholding)
        # Since we have only 6 features, we can do this efficiently
        # Initialize with OLS solution
        try:
            VtV = V_norm.T @ V_norm
            Vty = V_norm.T @ y_norm
            VtV_inv = np.linalg.inv(VtV + 1e-8 * np.eye(6))
            w_init = VtV_inv @ Vty
        except np.linalg.LinAlgError:
            w_init = np.zeros(6)
        
        # Coordinate descent for Lasso
        w = w_init.copy()
        for _ in range(50):
            for j in range(6):
                # Compute residual without feature j
                r = y_norm - V_norm @ w
                r += V_norm[:, j] * w[j]
                # Update feature j
                if np.abs(V_norm[:, j]).sum() < 1e-12:
                    w[j] = 0.0
                    continue
                rho = V_norm[:, j] @ r
                z = VtV[j, j]
                # Soft threshold
                if rho > alpha:
                    w_new = (rho - alpha) / z
                elif rho < -alpha:
                    w_new = (rho + alpha) / z
                else:
                    w_new = 0.0
                w[j] = w_new
        
        # Convert back to original scale
        # V_norm @ w = (V - col_means)/col_stds @ w
        # We want V @ coeff = V_norm @ w + col_means * w ... 
        # Actually: y = V_norm @ w + y_mean => y = (V - col_means)/col_stds @ w + y_mean
        # => y = V @ (w/col_stds) - col_means @ w + y_mean
        # So coeff in original polynomial space:
        # The fitted value is V @ c where c satisfies V @ c = (V - col_means)/col_stds @ w + y_mean
        # This is tricky. Instead, let's just fit directly in original space.
        
        # Alternative: use the L1-selected support and refit with OLS
        support = np.where(np.abs(w) > 1e-6)[0]
        if len(support) == 0:
            support = np.array([1])  # at least linear
        
        # Refit with OLS on selected features
        try:
            V_sub = V[:, support]
            VtV_sub = V_sub.T @ V_sub
            Vty_sub = V_sub.T @ y
            VtV_sub_inv = np.linalg.inv(VtV_sub + 1e-10 * np.eye(len(support)))
            c_sub = VtV_sub_inv @ Vty_sub
        except np.linalg.LinAlgError:
            c_sub = np.zeros(len(support))
        
        coeff = np.zeros(6)
        coeff[support] = c_sub
        
        # Compute training MSE
        pred = V @ coeff
        mse = np.mean((y - pred) ** 2)
        
        # Penalize complexity slightly to prefer simpler models if MSE is similar
        complexity_penalty = 0.01 * len(support)
        score = mse + complexity_penalty * y_std ** 2
        
        if score < best_score:
            best_score = score
            best_coeff = coeff
    
    # Final check: ensure coefficients are finite
    if not np.all(np.isfinite(best_coeff)):
        # Fallback to simple linear fit
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c_lin
        except Exception:
            best_coeff = np.zeros(6)
    
    return best_coeff
# EVOLVE-BLOCK-END
