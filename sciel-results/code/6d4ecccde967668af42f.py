import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    if n < 10:
        # Fallback: mean
        mean_y = np.mean(y) if n > 0 else 0.0
        return np.array([mean_y, 0, 0, 0, 0, 0], dtype=np.float64)
    
    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15
        
    # Scale x to [-1, 1] for better conditioning
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_min = x_min - 0.5
        x_max = x_min + 0.5
        
    # Linear map: x_scaled = A * x + B
    A = 2.0 / x_range
    B = - (x_max + x_min) / x_range
    x_scaled = A * x + B
    
    # Model selection via 5-Fold Cross Validation
    degrees = [0, 1, 2, 3, 4, 5]
    alphas = np.logspace(-6, 6, 60)
    
    k_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // k_folds
    folds = [indices[i*fold_size : (i+1)*fold_size] for i in range(k_folds)]
    if len(folds[-1]) < fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds*fold_size:]])
    
    best_cv_mse = np.inf
    best_degree = 1
    best_alpha = 1.0
    
    for deg in degrees:
        if time.time() > deadline:
            break
            
        V_all = np.vander(x_scaled, deg + 1, increasing=True)
        
        # Accumulate MSE for each alpha across folds
        alpha_mse_sums = np.zeros(len(alphas))
        
        for i in range(k_folds):
            if time.time() > deadline:
                break
                
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            V_train = V_all[train_idx]
            y_train = y[train_idx]
            V_val = V_all[val_idx]
            y_val = y[val_idx]
            
            # Precompute VtV and Vty for this fold
            VtV = V_train.T @ V_train
            Vty = V_train.T @ y_train
            d = deg + 1
            
            # Eigen-decompose VtV for fast multi-alpha ridge solves
            try:
                eigvals, eigvecs = np.linalg.eigh(VtV)
                eigvals = np.maximum(eigvals, 0.0)
            except np.linalg.LinAlgError:
                for j, alpha in enumerate(alphas):
                    try:
                        A_mat = VtV + alpha * np.eye(d)
                        theta = np.linalg.solve(A_mat, Vty)
                        y_pred = V_val @ theta
                        mse = np.mean((y_pred - y_val) ** 2)
                        alpha_mse_sums[j] += mse
                    except Exception:
                        continue
                continue
            
            # theta = eigvecs @ (eigvecs.T @ Vty / (eigvals + alpha))
            VtVty = eigvecs.T @ Vty
            
            for j, alpha in enumerate(alphas):
                scaled = VtVty / (eigvals + alpha)
                theta = eigvecs @ scaled
                y_pred = V_val @ theta
                mse = np.mean((y_pred - y_val) ** 2)
                alpha_mse_sums[j] += mse
        
        if time.time() > deadline:
            break
            
        min_idx = np.argmin(alpha_mse_sums)
        avg_cv_mse = alpha_mse_sums[min_idx] / k_folds
        
        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_degree = deg
            best_alpha = alphas[min_idx]
            
    # Final fit on all data with best_degree and best_alpha
    V_all = np.vander(x_scaled, best_degree + 1, increasing=True)
    reg_matrix = np.eye(best_degree + 1) * best_alpha
    A_mat = V_all.T @ V_all + reg_matrix
    b_vec = V_all.T @ y
    
    try:
        theta_scaled = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        theta_scaled = np.zeros(best_degree + 1)
        
    # Convert coefficients from scaled monomial basis to original monomial basis
    # P(x_scaled) = sum_{i=0}^{d} theta_scaled[i] * x_scaled^i
    # x_scaled = A * x + B
    # P(x) = sum_{i=0}^{d} theta_scaled[i] * (A * x + B)^i
    
    # We want coeffs c_0...c_5 for x^0...x^5
    # Start with zero polynomial
    result_poly = np.zeros(6, dtype=np.float64)
    
    # (A*x + B) is represented by coeffs [B, A] in ascending order
    base_poly = np.array([B, A], dtype=np.float64)
    
    current_power = np.array([1.0], dtype=np.float64) # (Ax+B)^0
    
    for i in range(best_degree + 1):
        # Add theta_scaled[i] * current_power to result_poly
        # Pad current_power to length 6
        if len(current_power) < 6:
            cp_padded = np.pad(current_power, (0, 6 - len(current_power)), mode='constant')
        else:
            cp_padded = current_power[:6]
            
        result_poly += theta_scaled[i] * cp_padded
        
        # Update current_power for next iteration
        if i < best_degree:
            current_power = np.polynomial.polynomial.polymul(current_power, base_poly)
            # Trim to max degree 5 to keep size small
            if len(current_power) > 6:
                current_power = current_power[:6]
                
    # Ensure finite
    if not np.all(np.isfinite(result_poly)):
        result_poly = np.array([np.mean(y), 0, 0, 0, 0, 0], dtype=np.float64)
        
    return result_poly
