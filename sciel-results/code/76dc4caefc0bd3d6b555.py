import numpy as np
import scipy.special as sp
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    # Filter non-finite
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    if n < 10:
        return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
        
    x_scaled = (x - mu) / sigma
    
    deg = 5
    
    # Check time
    if time.time() > deadline - 10.0:
        # Fallback: OLS linear
        coeff = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # Subsample for CV if large
    max_cv_samples = 60000
    if n > max_cv_samples:
        idx = np.random.choice(n, max_cv_samples, replace=False)
        x_cv = x_scaled[idx]
        y_cv = y[idx]
    else:
        x_cv = x_scaled
        y_cv = y
        
    n_cv = len(x_cv)
    
    V_cv = np.vander(x_cv, N=deg+1, increasing=True)
    
    # Coarser lambda grid to save time, then refine
    lambdas = np.logspace(-8, 8, 40)
    k = 5
    indices = np.arange(n_cv)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    best_lambda = 0.0
    min_cv_mse = np.inf
    
    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
        mse_sum = 0.0
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            V_train = V_cv[train_idx]
            y_train = y_cv[train_idx]
            V_val = V_cv[val_idx]
            y_val = y_cv[val_idx]
            
            reg_matrix = np.eye(deg+1)
            reg_matrix[0,0] = 0.0
            
            A = V_train.T @ V_train + lam * reg_matrix
            b = V_train.T @ y_train
            
            try:
                beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(A, b, rcond=None)[0]
                
            y_pred = V_val @ beta
            mse = np.mean((y_val - y_pred)**2)
            mse_sum += mse
            
        mean_mse = mse_sum / k
        if mean_mse < min_cv_mse:
            min_cv_mse = mean_mse
            best_lambda = lam
    
    # Refine around best_lambda with finer grid
    if best_lambda > 0:
        ratio = 1.5
        lam_low = best_lambda / ratio
        lam_high = best_lambda * ratio
        fine_lambdas = np.logspace(np.log10(lam_low), np.log10(lam_high), 20)
        
        for lam in fine_lambdas:
            if time.time() > deadline - 5.0:
                break
            mse_sum = 0.0
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                V_train = V_cv[train_idx]
                y_train = y_cv[train_idx]
                V_val = V_cv[val_idx]
                y_val = y_cv[val_idx]
                
                reg_matrix = np.eye(deg+1)
                reg_matrix[0,0] = 0.0
                
                A = V_train.T @ V_train + lam * reg_matrix
                b = V_train.T @ y_train
                
                try:
                    beta = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(A, b, rcond=None)[0]
                    
                y_pred = V_val @ beta
                mse = np.mean((y_val - y_pred)**2)
                mse_sum += mse
                
            mean_mse = mse_sum / k
            if mean_mse < min_cv_mse:
                min_cv_mse = mean_mse
                best_lambda = lam
    
    # Final Fit on all data
    V_full = np.vander(x_scaled, N=deg+1, increasing=True)
    reg_matrix = np.eye(deg+1)
    reg_matrix[0,0] = 0.0
    A = V_full.T @ V_full + best_lambda * reg_matrix
    b = V_full.T @ y
    
    try:
        a_scaled = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        a_scaled = np.linalg.lstsq(A, b, rcond=None)[0]
        
    # Convert coefficients
    c = np.zeros(deg+1)
    for j in range(deg+1):
        sum_val = 0.0
        for i in range(j, deg+1):
            binom_val = sp.comb(i, j, exact=True)
            term = a_scaled[i] * (sigma ** (-i)) * binom_val * ((-mu) ** (i-j))
            sum_val += term
        c[j] = sum_val
        
    return c
