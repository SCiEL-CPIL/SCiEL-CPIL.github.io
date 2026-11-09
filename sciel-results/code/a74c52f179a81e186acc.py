import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit (task says 120s, prompt says 150s kill, use 110 for safety)
    
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
    
    # If n is very small, just do a linear fit
    if n < 20:
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff

    # Define search space
    degrees = [1, 2, 3, 4, 5]
    # Log-spaced lambdas for ridge regularization
    # We need to tune lambda relative to the scale of the data.
    # A common heuristic is to start with a wide range.
    lambdas = [1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4]
    
    # K-Fold Cross Validation
    k = 5
    indices = np.arange(n)
    np.random.seed(42) # For reproducibility in CV splits
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    best_score = float('inf')
    best_degree = 1
    best_lambda = 0.0 # 0 means OLS
    
    # Helper to build Vandermonde matrix
    def vandermonde(x, deg):
        # x is 1D array, deg is max degree
        # Returns matrix of shape (len(x), deg+1)
        # Columns: x^0, x^1, ..., x^deg
        # Using np.vander with increasing=True
        return np.vander(x, N=deg+1, increasing=True)
    
    # Helper to solve ridge regression
    def solve_ridge(A, y, lam):
        # Solve (A^T A + lam I) c = A^T y
        # Using augmented system for stability:
        # [ A ] [ c ]   [ y ]
        # [ sqrt(lam) I ] [0] = [0]
        # But for lam=0, just lstsq
        if lam == 0:
            coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            return coeff
        else:
            sqrt_lam = np.sqrt(lam)
            m, p = A.shape
            A_aug = np.vstack([A, sqrt_lam * np.eye(p)])
            y_aug = np.concatenate([y, np.zeros(p)])
            coeff, _, _, _ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
            return coeff

    for deg in degrees:
        if time.time() > deadline:
            break
            
        for lam in lambdas:
            if time.time() > deadline:
                break
                
            cv_mses = []
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                x_train = x[train_idx]
                y_train = y[train_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]
                
                # Build training matrix
                A_train = vandermonde(x_train, deg)
                
                # Solve
                coeff = solve_ridge(A_train, y_train, lam)
                
                # Evaluate on validation
                A_val = vandermonde(x_val, deg)
                y_pred = A_val @ coeff
                mse = np.mean((y_val - y_pred) ** 2)
                cv_mses.append(mse)
            
            mean_mse = np.mean(cv_mses)
            
            if mean_mse < best_score:
                best_score = mean_mse
                best_degree = deg
                best_lambda = lam

    # Refit on entire training set with best parameters
    A_full = vandermonde(x, best_degree)
    final_coeff = solve_ridge(A_full, y, best_lambda)
    
    # Ensure output is 6 coefficients in ascending powers
    result = np.zeros(6, dtype=np.float64)
    result[:best_degree+1] = final_coeff
    
    # Check for non-finite results
    if not np.all(np.isfinite(result)):
        # Fallback to linear fit
        result = np.zeros(6, dtype=np.float64)
        result[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        if not np.all(np.isfinite(result)):
            result = np.zeros(6, dtype=np.float64)
            
    return result
