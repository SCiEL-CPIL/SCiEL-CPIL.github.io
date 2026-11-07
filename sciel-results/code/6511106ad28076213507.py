"""
Strategy: Adaptive Gaussian Process Regression with Multiple Kernels and Length Scales.
Rationale: 
1. The hidden equations are 1D, smooth, and noiseless (200 points).
2. Gaussian Processes (GPs) are non-parametric, flexible, and provide exact
   interpolation for noiseless data when the noise variance is set very small.
3. By using a combination of kernels (RBF + Linear + Constant), we can capture
   both smooth oscillatory behaviors (RBF) and linear trends (Linear).
4. We perform a grid search over length scales and signal variances to find the
   optimal hyperparameters via cross-validation. This is a structural change
   from the previous piecewise spline approach, moving to a global probabilistic
   model with adaptive hyperparameters.
"""
import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer for 300s limit
    
    X = X_train[:, 0]  # 1D input
    y = y_train
    
    n = len(X)
    x_min, x_max = X.min(), X.max()
    
    # Normalize input to [0, 1] for better GP conditioning
    x_norm = (X - x_min) / (x_max - x_min + 1e-10)
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / (y_std + 1e-10)
    
    # Sort data for efficiency
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_sorted = y_norm[sort_idx]
    
    # Define kernels
    def kernel_rbf(X1, X2, length_scale):
        # X1, X2 are 1D arrays
        sq_dists = (X1[:, None] - X2[None, :]) ** 2
        return np.exp(-0.5 * sq_dists / (length_scale ** 2))
    
    def kernel_linear(X1, X2):
        return X1[:, None] * X2[None, :]
    
    def kernel_constant(X1, X2):
        return np.ones((len(X1), len(X2)))
    
    def compute_kernel_matrix(x1, x2, length_scale, signal_var, linear_var):
        K = signal_var * kernel_rbf(x1, x2, length_scale)
        K += linear_var * kernel_linear(x1, x2)
        K += 0.1 * kernel_constant(x1, x2)  # Small constant to ensure invertibility
        return K
    
    def log_marginal_likelihood(hyperparams, x, y):
        length_scale, signal_var, linear_var = hyperparams
        # Ensure positive values
        length_scale = max(length_scale, 1e-3)
        signal_var = max(signal_var, 1e-3)
        linear_var = max(linear_var, 1e-3)
        
        n = len(x)
        K = compute_kernel_matrix(x, x, length_scale, signal_var, linear_var)
        K += 1e-6 * np.eye(n)  # Noise term
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return -1e10
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        return lml
    
    def negative_log_marginal_likelihood(hyperparams, x, y):
        return -log_marginal_likelihood(hyperparams, x, y)
    
    # Cross-validation setup
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    folds[-1] = np.concatenate([folds[-1], indices[n_folds * fold_size:]])  # Handle remainder
    
    # Hyperparameter grid search
    length_scale_range = np.logspace(-1, 1, 10)  # 0.1 to 10
    signal_var_range = np.logspace(-2, 2, 5)     # 0.01 to 100
    linear_var_range = np.logspace(-3, 1, 4)     # 0.001 to 10
    
    best_score = np.inf
    best_params = None
    best_predict = None
    
    # We'll do a coarse grid search first, then refine if time permits
    # But for simplicity and robustness, let's just do a reasonable grid.
    # Given 200 points, we can afford a few hundred GP fits.
    
    param_grid = []
    for ls in length_scale_range:
        for sv in signal_var_range:
            for lv in linear_var_range:
                param_grid.append((ls, sv, lv))
    
    # Shuffle grid to explore diverse regions early
    np.random.shuffle(param_grid)
    
    # Limit grid size to avoid timeout
    max_params_to_try = 200
    param_grid = param_grid[:max_params_to_try]
    
    for i, (ls, sv, lv) in enumerate(param_grid):
        if time.time() > deadline:
            break
            
        try:
            cv_errors = []
            for k in range(n_folds):
                val_idx = folds[k]
                train_idx = np.setdiff1d(indices, val_idx)
                
                x_val = x_sorted[val_idx]
                y_val = y_sorted[val_idx]
                x_tr = x_sorted[train_idx]
                y_tr = y_sorted[train_idx]
                
                # Fit GP on training data
                K_tr = compute_kernel_matrix(x_tr, x_tr, ls, sv, lv)
                K_tr += 1e-6 * np.eye(len(x_tr))
                
                try:
                    L_tr = np.linalg.cholesky(K_tr)
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
                    continue
                
                alpha_tr = np.linalg.solve(L_tr.T, np.linalg.solve(L_tr, y_tr))
                
                # Predict on validation data
                K_val_tr = compute_kernel_matrix(x_val, x_tr, ls, sv, lv)
                pred_norm = K_val_tr @ alpha_tr
                pred = pred_norm * y_std + y_mean
                
                mse = np.mean((pred - y[val_idx]) ** 2)
                cv_errors.append(mse)
            
            mean_mse = np.mean(cv_errors)
            if not np.isfinite(mean_mse):
                continue
                
            if mean_mse < best_score:
                best_score = mean_mse
                best_params = (ls, sv, lv)
                
                # Refit on full data
                K_full = compute_kernel_matrix(x_sorted, x_sorted, ls, sv, lv)
                K_full += 1e-6 * np.eye(n)
                
                try:
                    L_full = np.linalg.cholesky(K_full)
                except np.linalg.LinAlgError:
                    continue
                
                alpha_full = np.linalg.solve(L_full.T, np.linalg.solve(L_full, y_sorted))
                
                def predict(X, x_sorted=x_sorted, K_full=K_full, L_full=L_full, alpha_full=alpha_full, 
                            x_min=x_min, x_max=x_max, y_mean=y_mean, y_std=y_std,
                            ls=ls, sv=sv, lv=lv):
                    X_flat = X.flatten()
                    x_pred_norm = (X_flat - x_min) / (x_max - x_min + 1e-10)
                    
                    # Compute covariance between test points and training points
                    K_test_tr = compute_kernel_matrix(x_pred_norm, x_sorted, ls, sv, lv)
                    pred_norm = K_test_tr @ alpha_full
                    pred = pred_norm * y_std + y_mean
                    
                    return pred.reshape(X.shape)
                
                best_predict = predict
                
        except Exception:
            continue
    
    if best_predict is None:
        # Fallback: Use a simple RBF kernel with default parameters
        try:
            ls, sv, lv = 0.5, 1.0, 0.1
            K_full = compute_kernel_matrix(x_sorted, x_sorted, ls, sv, lv)
            K_full += 1e-6 * np.eye(n)
            L_full = np.linalg.cholesky(K_full)
            alpha_full = np.linalg.solve(L_full.T, np.linalg.solve(L_full, y_sorted))
            
            def predict(X, x_sorted=x_sorted, x_min=x_min, x_max=x_max, y_mean=y_mean, y_std=y_std,
                        ls=ls, sv=sv, lv=lv, alpha_full=alpha_full):
                X_flat = X.flatten()
                x_pred_norm = (X_flat - x_min) / (x_max - x_min + 1e-10)
                K_test_tr = compute_kernel_matrix(x_pred_norm, x_sorted, ls, sv, lv)
                pred_norm = K_test_tr @ alpha_full
                pred = pred_norm * y_std + y_mean
                return pred.reshape(X.shape)
            
            return predict
        except Exception:
            mean_y = np.mean(y)
            return lambda X: np.full(X.shape[0], mean_y)
    
    return best_predict
