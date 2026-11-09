import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
        
    x_norm = (x - x_mean) / x_std
    
    # Split for model selection
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_size = max(100, min(n // 10, 5000)) # Cap validation size for speed
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train = x_norm[train_idx]
    y_train = y[train_idx]
    x_val = x_norm[val_idx]
    y_val = y[val_idx]
    
    degree = 5
    V_train = np.vander(x_train, N=degree + 1, increasing=True)
    V_val = np.vander(x_val, N=degree + 1, increasing=True)
    
    best_score = np.inf
    best_lam = 1e-4
    
    # Try a range of regularization strengths
    lambdas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    
    for lam in lambdas:
        if time.time() - t_start > deadline - 5:
            break
            
        # Ridge Regression: (V^T V + lam I) c = V^T y
        A = V_train.T @ V_train + lam * np.eye(degree + 1)
        b = V_train.T @ y_train
        
        try:
            coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            
        if not np.all(np.isfinite(coeff)):
            continue
            
        pred_val = V_val @ coeff
        val_mse = np.mean((pred_val - y_val) ** 2)
        
        if val_mse < best_score:
            best_score = val_mse
            best_lam = lam
            
    # Refit on ALL data with best lambda
    V_all = np.vander(x_norm, N=degree + 1, increasing=True)
    A_all = V_all.T @ V_all + best_lam * np.eye(degree + 1)
    b_all = V_all.T @ y
    
    try:
        coeff_norm = np.linalg.solve(A_all, b_all)
    except np.linalg.LinAlgError:
        coeff_norm = np.linalg.lstsq(A_all, b_all, rcond=None)[0]
        
    if not np.all(np.isfinite(coeff_norm)):
        coeff_norm = np.zeros(degree + 1)
        
    # Convert from normalized x to original x scale
    # coeff_norm corresponds to sum c_i * x_norm^i
    # x_norm = (x - x_mean) / x_std
    # We need coeff_orig such that sum c_orig_j * x^j = sum c_norm_i * ((x - x_mean) / x_std)^i
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for i in range(degree + 1):
        ci = coeff_norm[i]
        if abs(ci) < 1e-15:
            continue
            
        # Term: ci * ( (x - x_mean) / x_std )^i = ci / x_std^i * (x - x_mean)^i
        ci_scaled = ci / (x_std ** i)
        
        # Expand (x - x_mean)^i = sum_{j=0}^i binom(i, j) x^j (-x_mean)^(i-j)
        for j in range(i + 1):
            # Binomial coefficient C(i, j)
            binom_val = 1.0
            for k in range(j):
                binom_val = binom_val * (i - k) / (k + 1)
                
            term = binom_val * ((-x_mean) ** (i - j)) * ci_scaled
            coeff_orig[j] += term
            
    # Ensure finite
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.nan_to_num(coeff_orig, nan=0.0, posinf=0.0, neginf=0.0)
        
    return coeff_orig
