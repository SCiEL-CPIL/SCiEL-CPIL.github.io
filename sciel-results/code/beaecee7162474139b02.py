import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    N = X_train.shape[0]
    
    # Pre-compute squared distances for RBF kernel
    # dist^2 = sum(x_i^2) + sum(x_j^2) - 2 x_i . x_j
    X_sq = np.sum(X_train**2, axis=1)
    dist_sq = X_sq[:, None] + X_sq[None, :] - 2 * X_train @ X_train.T
    np.maximum(dist_sq, 0, out=dist_sq) # Numerical stability
    
    best_gamma = 1.0
    best_lambda = 1e-3
    best_loocv_mse = float('inf')
    
    # Grid search
    gammas = np.logspace(-3, 2, 10)
    lambdas = np.logspace(-4, 0, 5)
    
    for gamma in gammas:
        K = np.exp(-gamma * dist_sq)
        for lam in lambdas:
            A = K + lam * np.eye(N)
            try:
                w = np.linalg.solve(A, y_train)
            except np.linalg.LinAlgError:
                continue
                
            # LOOCV calculation
            # The diagonal of A^{-1} is needed.
            # We can compute A_inv_diag efficiently or just use the formula.
            # Actually, calculating the full inverse is O(N^3) which we already did in solve?
            # No, solve gives w. To get (A^{-1})_ii, we need the inverse.
            # Alternative: Compute LOOCV predictions directly?
            # \hat{y}_{-i} = y_i - w_i / (A^{-1})_{ii}
            # We can compute A_inv = np.linalg.inv(A) which is O(N^3).
            # Since we are in a loop, this might be heavy but N=200 is small.
            # Let's try to be efficient.
            
            # A simpler and robust way for small N:
            # Just compute the full inverse once per (gamma, lambda) pair?
            # Or use the formula:
            # A^{-1} = (K + \lambda I)^{-1}
            # This is expensive.
            
            # Let's stick to the direct LOOCV formula using the inverse diagonal.
            # We can compute the inverse diagonal using a trick or just invert.
            # Given N=200, np.linalg.inv is fast enough.
            
            A_inv = np.linalg.inv(A)
            diag_inv = np.diag(A_inv)
            
            # LOOCV predictions
            y_pred_loocv = y_train - w / diag_inv
            
            mse = np.mean((y_pred_loocv - y_train)**2)
            
            if mse < best_loocv_mse:
                best_loocv_mse = mse
                best_gamma = gamma
                best_lambda = lam
                best_w = w
                best_K = K

    # Final prediction function
    def predict(X_test):
        # Compute kernel between test and train
        # dist_sq_test = sum(x_t^2) + sum(x_i^2) - 2 x_t . x_i
        X_test_sq = np.sum(X_test**2, axis=1)
        dist_sq_test = X_test_sq[:, None] + np.sum(X_train**2, axis=1)[None, :] - 2 * X_test @ X_train.T
        np.maximum(dist_sq_test, 0, out=dist_sq_test)
        
        K_test = np.exp(-best_gamma * dist_sq_test)
        return K_test @ best_w

    return predict
