# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import polynomial as P


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale
    forcing of a two-scale Lorenz-96 model using robust weighted least squares
    with adaptive polynomial degree selection and regularization.
    
    Args:
        training_x: array-like, flattened slow-state values
        training_subgrid_tendency: array-like, flattened subgrid tendency values
    
    Returns:
        np.ndarray of shape (6,) with coefficients in ASCENDING powers of x
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Compute statistics for scaling
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    
    # Scale x to [-1, 1] range approximately for numerical stability
    x_scaled = (x - x_mean) / x_std
    
    # Try multiple regularization strengths and select best via validation-like criterion
    # We'll use a simple hold-out approach: split into train/val
    n = len(x_scaled)
    
    # Shuffle for consistent splitting
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    
    # Use 80/20 split for model selection
    val_frac = 0.2
    n_val = max(1, int(n * val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    x_train = x_scaled[train_idx]
    y_train = y[train_idx]
    x_val = x_scaled[val_idx]
    y_val = y[val_idx]
    
    # Try different regularization strengths and polynomial degrees
    # We always return degree 5, but we can use regularization to effectively lower the degree
    
    best_score = np.inf
    best_coeff = np.zeros(6)
    
    # Regularization strengths to try
    lambdas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # Build Vandermonde matrix for degree 5
    def make_vandermonde(x_vals, deg=5):
        """Create Vandermonde matrix with columns [1, x, x^2, ..., x^deg]"""
        powers = np.arange(deg + 1)
        return x_vals[:, None] ** powers
    
    # For each lambda, fit and evaluate
    for lam in lambdas:
        V_train = make_vandermonde(x_train, 5)
        V_val = make_vandermonde(x_val, 5)
        
        # Add regularization: (V^T V + lam * I) c = V^T y
        # Use identity for regularization, but we might want to regularize higher degrees more
        # Let's use a diagonal regularization that increases with degree
        I = np.eye(6)
        # Weighted regularization: higher degrees get more penalty
        reg_weights = np.array([1.0, 1.0, 5.0, 10.0, 50.0, 100.0])
        reg_matrix = lam * np.diag(reg_weights)
        
        try:
            A = V_train.T @ V_train + reg_matrix
            b = V_train.T @ y_train
            
            # Solve the linear system
            coeff_scaled = np.linalg.solve(A, b)
            
            # Evaluate on validation set
            y_pred_val = V_val @ coeff_scaled
            mse_val = np.mean((y_pred_val - y_val) ** 2)
            
            # Also compute normalized MSE (divide by variance of y)
            y_var = np.var(y_val)
            if y_var < 1e-12:
                y_var = 1.0
            norm_mse = mse_val / y_var
            
            if norm_mse < best_score:
                best_score = norm_mse
                best_coeff_scaled = coeff_scaled.copy()
        
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            try:
                coeff_scaled, _, _, _ = np.linalg.lstsq(V_train, y_train, rcond=None)
                y_pred_val = V_val @ coeff_scaled
                mse_val = np.mean((y_pred_val - y_val) ** 2)
                y_var = np.var(y_val)
                if y_var < 1e-12:
                    y_var = 1.0
                norm_mse = mse_val / y_var
                
                if norm_mse < best_score:
                    best_score = norm_mse
                    best_coeff_scaled = coeff_scaled.copy()
            except:
                continue
    
    # Now refit on ALL data with the best lambda
    # Find the best lambda
    best_lambda = lambdas[0]
    best_score_refit = np.inf
    best_coeff_final = np.zeros(6)
    
    for lam in lambdas:
        V_all = make_vandermonde(x_scaled, 5)
        
        I = np.eye(6)
        reg_weights = np.array([1.0, 1.0, 5.0, 10.0, 50.0, 100.0])
        reg_matrix = lam * np.diag(reg_weights)
        
        try:
            A = V_all.T @ V_all + reg_matrix
            b = V_all.T @ y
            
            coeff_scaled = np.linalg.solve(A, b)
            
            # Use the validation score from before as proxy
            # Actually, let's just pick the lambda that gave best validation score
            # We already have best_coeff_scaled from above, so use that
            break
            
        except:
            continue
    
    # Use the best coefficient found during validation
    # But we need to convert from scaled x back to original x
    # If x_scaled = (x - x_mean) / x_std, then x = x_scaled * x_std + x_mean
    # So if we have polynomial in x_scaled: sum(c_i * x_scaled^i)
    # We need polynomial in x: sum(d_i * x^i)
    
    # x_scaled = (x - x_mean) / x_std
    # x = x_scaled * x_std + x_mean
    
    # Let's convert using polynomial operations
    # c(x_scaled) = sum_{i=0}^{5} c_i * x_scaled^i
    # x_scaled = (x - x_mean) / x_std
    # So c(x) = sum_{i=0}^{5} c_i * ((x - x_mean) / x_std)^i
    #         = sum_{i=0}^{5} c_i / x_std^i * (x - x_mean)^i
    
    # We need to expand (x - x_mean)^i in terms of x^j
    # (x - a)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-a)^(i-j)
    
    from math import comb
    
    coeff_original = np.zeros(6)
    for i in range(6):
        c_i = best_coeff_scaled[i]
        if abs(c_i) < 1e-15:
            continue
        # Expand (x - x_mean)^i
        for j in range(i + 1):
            coeff_original[j] += c_i / (x_std ** i) * comb(i, j) * ((-x_mean) ** (i - j))
    
    # Clip to finite values
    coeff_original = np.where(np.isfinite(coeff_original), coeff_original, 0.0)
    
    return coeff_original
# EVOLVE-BLOCK-END
