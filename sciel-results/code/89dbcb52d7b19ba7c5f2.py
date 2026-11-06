"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n_samples = len(x)
    if n_samples < 10:
        return np.zeros(6)
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Use k-fold cross-validation for model selection (5-fold)
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n_samples // n_folds
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    if n_samples % n_folds != 0:
        folds[-1] = np.concatenate([folds[-1], indices[(n_folds-1)*fold_size:]])
    
    best_mse = np.inf
    best_degree = 1
    best_lambda = 1e-4
    best_cv_scores = []
    
    # Regularization strengths - finer grid
    # Use log-spaced values from very small to large
    lambdas = np.logspace(-10, 3, 20)
    # Add some specific values
    lambdas = np.unique(np.concatenate([lambdas, [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]]))
    
    for degree in range(1, 6):
        if time.time() > deadline - 5.0:
            break
        
        # Build Vandermonde matrices for all folds
        # V[:, i] = x_scaled ** i
        V_all = np.vander(x_scaled, N=degree + 1, increasing=True)
        
        fold_mses = []
        valid = True
        
        for fold_i in range(n_folds):
            if time.time() > deadline - 5.0:
                valid = False
                break
            
            val_idx = folds[fold_i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])
            
            V_train = V_all[train_idx]
            V_val = V_all[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            for lam in lambdas:
                if time.time() > deadline - 5.0:
                    valid = False
                    break
                
                # Ridge regression: (V^T V + lambda * I) c = V^T y
                reg_matrix = V_train.T @ V_train
                if lam > 0:
                    reg_matrix = reg_matrix + lam * np.eye(degree + 1)
                else:
                    reg_matrix = reg_matrix + 1e-12 * np.eye(degree + 1)
                
                rhs = V_train.T @ y_train
                
                try:
                    coeff_candidate = np.linalg.solve(reg_matrix, rhs)
                    pred_val = V_val @ coeff_candidate
                    mse_val = np.mean((pred_val - y_val) ** 2)
                    
                    if lam == 0 or mse_val < best_mse:
                        # Track best per fold
                        pass
                except np.linalg.LinAlgError:
                    continue
        
        if not valid:
            break
        
        # Recompute: for each (degree, lambda), compute mean CV MSE
        # This is expensive, so let's do it more efficiently
        # Actually, let's restructure: for each degree, find best lambda via CV
    
    # Better approach: for each degree, find best lambda via CV
    best_mse = np.inf
    best_degree = 1
    best_lambda = 1e-4
    
    for degree in range(1, 6):
        if time.time() > deadline - 10.0:
            break
        
        V_all = np.vander(x_scaled, N=degree + 1, increasing=True)
        
        best_lambda_for_degree = 1e-4
        best_mse_for_degree = np.inf
        
        for lam in lambdas:
            if time.time() > deadline - 10.0:
                break
            
            fold_mses = []
            for fold_i in range(n_folds):
                val_idx = folds[fold_i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])
                
                V_train = V_all[train_idx]
                V_val = V_all[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]
                
                reg_matrix = V_train.T @ V_train
                if lam > 0:
                    reg_matrix = reg_matrix + lam * np.eye(degree + 1)
                else:
                    reg_matrix = reg_matrix + 1e-12 * np.eye(degree + 1)
                
                rhs = V_train.T @ y_train
                
                try:
                    coeff_candidate = np.linalg.solve(reg_matrix, rhs)
                    pred_val = V_val @ coeff_candidate
                    mse_val = np.mean((pred_val - y_val) ** 2)
                    fold_mses.append(mse_val)
                except np.linalg.LinAlgError:
                    fold_mses.append(np.inf)
                    break
            
            if len(fold_mses) < n_folds:
                continue
                
            mean_cv_mse = np.mean(fold_mses)
            
            if mean_cv_mse < best_mse_for_degree:
                best_mse_for_degree = mean_cv_mse
                best_lambda_for_degree = lam
        
        if best_mse_for_degree < best_mse:
            best_mse = best_mse_for_degree
            best_degree = degree
            best_lambda = best_lambda_for_degree
    
    # Refit with the best degree and lambda on the FULL training data
    if time.time() < deadline:
        V_full = np.vander(x_scaled, N=best_degree + 1, increasing=True)
        reg_matrix = V_full.T @ V_full
        if best_lambda > 0:
            reg_matrix = reg_matrix + best_lambda * np.eye(best_degree + 1)
        else:
            reg_matrix = reg_matrix + 1e-12 * np.eye(best_degree + 1)
        
        rhs = V_full.T @ y
        
        try:
            coeff_scaled = np.linalg.solve(reg_matrix, rhs)
        except np.linalg.LinAlgError:
            # Fallback to linear fit
            coeff_scaled = np.zeros(best_degree + 1)
            if best_degree >= 1:
                coeff_scaled[:2] = np.polynomial.polynomial.polyfit(x_scaled, y, 1)
            else:
                coeff_scaled[0] = np.mean(y)
        
        # Transform back from scaled to original x
        # We have: y ≈ sum_{i=0}^{d} c_i * ((x - x_mean)/x_std)^i
        # = sum_{i=0}^{d} c_i * x_std^{-i} * (x - x_mean)^i
        # = sum_{i=0}^{d} c_i * x_std^{-i} * sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^{i-j}
        # = sum_{j=0}^{d} [sum_{i=j}^{d} c_i * x_std^{-i} * binom(i,j) * (-x_mean)^{i-j}] * x^j
        
        d = best_degree
        result = np.zeros(6)
        for j in range(d + 1):
            coeff_j = 0.0
            for i in range(j, d + 1):
                binom = 1.0
                # Compute binom(i, j)
                if i > 0:
                    # binom(i, j) = i! / (j! * (i-j)!)
                    # Use multiplicative formula for stability
                    if j == 0:
                        binom = 1.0
                    elif j == i:
                        binom = 1.0
                    else:
                        # binom(i, j) = prod_{k=1}^{j} (i - j + k) / k
                        binom = 1.0
                        for k in range(1, j + 1):
                            binom *= (i - j + k) / k
                else:
                    binom = 1.0
                
                coeff_j += coeff_scaled[i] * (x_std ** (-i)) * binom * ((-x_mean) ** (i - j))
            result[j] = coeff_j
        
        return result
    else:
        # Time out - return a reasonable default
        # Do a quick linear fit on scaled data
        try:
            V_full = np.vander(x_scaled, N=2, increasing=True)
            reg_matrix = V_full.T @ V_full + 1e-6 * np.eye(2)
            rhs = V_full.T @ y
            coeff_scaled = np.linalg.solve(reg_matrix, rhs)
            
            # Transform back
            result = np.zeros(6)
            for j in range(2):
                coeff_j = 0.0
                for i in range(j, 2):
                    if i == j:
                        binom = 1.0
                    else:
                        binom = float(i)
                    coeff_j += coeff_scaled[i] * (x_std ** (-i)) * binom * ((-x_mean) ** (i - j))
                result[j] = coeff_j
            return result
        except:
            return np.zeros(6)
# EVOLVE-BLOCK-END
