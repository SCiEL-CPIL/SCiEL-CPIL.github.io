import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    X_flat = X_train.flatten()
    y_flat = y_train.flatten()
    N = len(X_flat)
    
    # Reshape for kernel computation (N, 1)
    X_col = X_flat.reshape(-1, 1)
    
    def gaussian_kernel(X1, X2, gamma):
        # X1: (N, 1), X2: (M, 1)
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-gamma * sq_dists)

    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    
    best_val_mse = np.inf
    best_gamma = 1.0
    best_alpha = 1e-4
    
    # Use a fixed validation split for hyperparameter selection
    # Shuffle is not strictly necessary if the data is random, but good practice.
    # However, we must ensure the split is consistent.
    indices = np.arange(N)
    # Simple split: last 20% for validation
    val_size = int(0.2 * N)
    if val_size < 5: # Safety for small N, though N=200
        val_size = 5
    val_idx = indices[-val_size:]
    train_idx = indices[:-val_size]
    
    X_val = X_col[val_idx]
    y_val = y_flat[val_idx]
    X_tr = X_col[train_idx]
    y_tr = y_flat[train_idx]
    
    for gamma in gammas:
        for alpha in alphas:
            if time.time() - start_time > 280:
                break
            
            try:
                K_tr = gaussian_kernel(X_tr, X_tr, gamma)
                A = K_tr + alpha * np.eye(len(X_tr))
                w = np.linalg.solve(A, y_tr)
                
                K_val = gaussian_kernel(X_val, X_tr, gamma)
                pred_val = K_val @ w
                mse = np.mean((pred_val - y_val)**2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_gamma = gamma
                    best_alpha = alpha
            except Exception:
                continue

    # Final fit on all data
    K_full = gaussian_kernel(X_col, X_col, best_gamma)
    A_full = K_full + best_alpha * np.eye(N)
    try:
        w_final = np.linalg.solve(A_full, y_flat)
    except Exception:
        # Fallback: Mean
        mean_y = np.mean(y_flat)
        return lambda X: np.full_like(X.flatten(), mean_y).reshape(X.shape)
    
    def predict(X):
        X_eval = np.asarray(X).flatten().reshape(-1, 1)
        K_test = gaussian_kernel(X_eval, X_col, best_gamma)
        y_pred = K_test @ w_final
        return y_pred.reshape(np.asarray(X).shape)

    return predict
