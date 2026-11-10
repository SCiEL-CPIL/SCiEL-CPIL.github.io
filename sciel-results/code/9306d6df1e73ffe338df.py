import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
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
    # Degrees to consider: 1 to 5
    # Alphas: log spaced
    degrees = [1, 2, 3, 4, 5]
    alphas = np.logspace(-5, 5, 50)
    
    k_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_size = n // k_folds
    folds = [indices[i*fold_size : (i+1)*fold_size] for i in range(k_folds)]
    # Ensure last fold has remaining samples
    if len(folds[-1]) < fold_size:
        folds[-1] = np.concatenate([folds[-1], indices[k_folds*fold_size:]])
    
    best_cv_mse = np.inf
    best_degree = 1
    best_alpha = 1.0
    
    # Precompute Vandermonde matrices for each degree and fold?
    # Or just compute on the fly. n is likely not huge (flattened trajectories).
    # If n is large (e.g., 100k), building V for all folds might be costly but still fast.
    
    for deg in degrees:
        if time.time() > deadline:
            break
            
        # Build V for all data in monomial basis of x_scaled
        # V[i, j] = x_scaled[i] ** j
        V_all = np.vander(x_scaled, deg + 1, increasing=True)
        
        cv_mse_sum = 0.0
        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            V_train = V_all[train_idx]
            y_train = y[train_idx]
            V_val = V_all[val_idx]
            y_val = y[val_idx]
            
            for alpha in alphas:
                if time.time() > deadline:
                    break
                
                try:
                    # Ridge Regression: (V^T V + alpha I) theta = V^T y
                    # Note: alpha scales the identity. 
                    # Standard ridge: min ||V theta - y||^2 + alpha ||theta||^2
                    # Solution: (V^T V + alpha I) theta = V^T y
                    
                    # For numerical stability, we can solve the linear system
                    # Or use np.linalg.lstsq with regularization?
                    # Direct solve is fine for small degrees.
                    
                    reg_matrix = np.eye(deg + 1) * alpha
                    A_mat = V_train.T @ V_train + reg_matrix
                    b_vec = V_train.T @ y_train
                    
                    theta = np.linalg.solve(A_mat, b_vec)
                    
                    y_pred = V_val @ theta
                    mse = np.mean((y_pred - y_val) ** 2)
                    cv_mse_sum += mse
                    
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue
        
        avg_cv_mse = cv_mse_sum / k_folds
        
        if avg_cv_mse < best_cv_mse:
            best_cv_mse = avg_cv_mse
            best_degree = deg
            # We need to store the best alpha for this degree, or re-find it?
            # The loop structure above doesn't track which alpha gave the min for the *sum*.
            # We should track the min alpha per degree, or just re-run the selection for the best degree?
            # Better: track best alpha inside the degree loop.
            
    # The logic above has a flaw: it doesn't track which alpha was best for the degree.
    # Let's restructure to track best (degree, alpha) globally.
    
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
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            V_train = V_all[train_idx]
            y_train = y[train_idx]
            V_val = V_all[val_idx]
            y_val = y[val_idx]
            
            for j, alpha in enumerate(alphas):
                if time.time() > deadline:
                    break
                
                try:
                    reg_matrix = np.eye(deg + 1) * alpha
                    A_mat = V_train.T @ V_train + reg_matrix
                    b_vec = V_train.T @ y_train
                    
                    theta = np.linalg.solve(A_mat, b_vec)
                    
                    y_pred = V_val @ theta
                    mse = np.mean((y_pred - y_val) ** 2)
                    alpha_mse_sums[j] += mse
                    
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue
            
        if time.time() > deadline:
            break
            
        # Find best alpha for this degree
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
