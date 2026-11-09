"""Symbolic regression via Kernel Ridge Regression with composite kernels."""
import numpy as np
import time
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # safety margin before 300s kill
    
    n_samples, n_features = X_train.shape
    
    # Standardize inputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    # Standardize outputs
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_std_val = float(y_std) if y_std > 1e-8 else 1.0
    y_norm = (y_train - y_mean) / y_std_val
    
    def compute_rbf_kernel(X1, X2, length_scales):
        """Compute RBF kernel with independent length scales."""
        # length_scales should be an array of shape (n_features,)
        scaled_X1 = X1 / length_scales
        scaled_X2 = X2 / length_scales
        sq_dists = np.sum(scaled_X1**2, axis=1)[:, None] + np.sum(scaled_X2**2, axis=1)[None, :] - 2 * (scaled_X1 @ scaled_X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-0.5 * sq_dists)
    
    def compute_poly_kernel(X1, X2, degree, coef0):
        """Compute polynomial kernel."""
        return (coef0 + X1 @ X2.T) ** degree
    
    def compute_periodic_kernel(X1, X2, period):
        """Compute periodic kernel."""
        # X1, X2 are standardized
        # Use the first feature or a combination? Let's use all features scaled by period
        # Standard periodic kernel: exp(-2 * sin^2(pi * |x-y| / p))
        # For multi-dim, we can sum over dims or use distance
        # Let's use Euclidean distance for simplicity in multi-dim
        dists = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
        dists = np.maximum(dists, 0)
        return np.exp(-2.0 * np.sin(np.pi * dists / period)**2)
    
    def compute_composite_kernel(X1, X2, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period):
        """Compute composite kernel: rbf_var * RBF + poly_var * Poly + per_var * Periodic + Identity"""
        K_rbf = rbf_var * compute_rbf_kernel(X1, X2, rbf_ls)
        K_poly = poly_var * compute_poly_kernel(X1, X2, poly_deg, 1.0)
        K_per = per_var * compute_periodic_kernel(X1, X2, per_period)
        
        if X1 is X2:
            K = K_rbf + K_poly + K_per
        else:
            K = K_rbf + K_poly + K_per
        return K
    
    def ridge_fit_predict(X_train_norm, y_train_norm, X_test_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha):
        """Fit KRR and predict."""
        K_train = compute_composite_kernel(X_train_norm, X_train_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        K_train = K_train + alpha * np.eye(n_samples)
        
        try:
            # Solve (K + alpha*I) w = y
            # Use Cholesky for stability
            L = np.linalg.cholesky(K_train)
            w = np.linalg.solve(L.T, np.linalg.solve(L, y_train_norm))
        except np.linalg.LinAlgError:
            try:
                w = np.linalg.solve(K_train, y_train_norm)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K_train, y_train_norm, rcond=None)[0]
        
        K_test = compute_composite_kernel(X_test_norm, X_train_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        pred_norm = K_test @ w
        return pred_norm * y_std_val + y_mean
    
    def loocv_score(X_norm, y_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha):
        """Compute LOOCV score (MSE)."""
        K_train = compute_composite_kernel(X_norm, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        K_reg = K_train + alpha * np.eye(n_samples)
        
        try:
            L = np.linalg.cholesky(K_reg)
            alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        except np.linalg.LinAlgError:
            try:
                alpha_vec = np.linalg.solve(K_reg, y_norm)
            except np.linalg.LinAlgError:
                alpha_vec = np.linalg.lstsq(K_reg, y_norm, rcond=None)[0]
        
        # LOOCV predictions: y_hat_i = y_i - K_ii / (K_reg^-1)_{ii}
        # (K_reg^-1)_{ii} = 1 / L_ii^2
        diag_L = np.diag(L)
        K_ii = np.diag(K_train)
        
        pred_i = y_norm - K_ii / (diag_L**2)
        errors = y_norm - pred_i
        mse = np.mean(errors**2)
        return mse
    
    best_score = np.inf
    best_params = None
    
    # Random search with independent length scales
    n_evals = 1500  # Slightly more evaluations to explore the larger space
    rng = np.random.RandomState(42)
    
    for i in range(n_evals):
        if time.time() > deadline - 10:
            break
            
        # Sample hyperparameters
        # Independent length scales for each feature
        ls = 10 ** rng.uniform(-1, 1, size=n_features)  # 0.1 to 10 per feature
        rbf_var = 10 ** rng.uniform(-1, 2)  # 0.1 to 100
        poly_deg = int(rng.choice([1, 2, 3, 4]))  # Increased max degree
        poly_var = 10 ** rng.uniform(-2, 1)  # 0.01 to 10
        per_var = 10 ** rng.uniform(-2, 1)  # 0.01 to 10
        per_period = 10 ** rng.uniform(0, 1.5)  # 1 to ~30
        alpha = 10 ** rng.uniform(-4, 2)  # 1e-4 to 100
        
        try:
            score = loocv_score(X_norm, y_norm, ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha)
            if score < best_score:
                best_score = score
                best_params = (ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha)
        except Exception:
            continue
    
    # If no valid params found, use defaults
    if best_params is None:
        best_params = (np.ones(n_features), 1.0, 2, 1.0, 1.0, 5.0, 1.0)
    
    rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha = best_params
    
    # Optional: A quick local refinement around the best point found
    # This helps if the random search landed near the optimum but not exactly on it
    if best_score < np.inf and time.time() < deadline - 20:
        local_rng = np.random.RandomState(43)
        best_ls = best_params[0].copy()
        best_rbf_var = best_params[1]
        best_poly_deg = best_params[2]
        best_poly_var = best_params[3]
        best_per_var = best_params[4]
        best_per_period = best_params[5]
        best_alpha = best_params[6]
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            # Perturb length scales
            new_ls = best_ls * 10 ** local_rng.uniform(-0.2, 0.2, size=n_features)
            new_rbf_var = best_rbf_var * 10 ** local_rng.uniform(-0.2, 0.2)
            new_poly_var = best_poly_var * 10 ** local_rng.uniform(-0.2, 0.2)
            new_per_var = best_per_var * 10 ** local_rng.uniform(-0.2, 0.2)
            new_per_period = best_per_period * 10 ** local_rng.uniform(-0.2, 0.2)
            new_alpha = best_alpha * 10 ** local_rng.uniform(-0.2, 0.2)
            
            try:
                score = loocv_score(X_norm, y_norm, new_ls, new_rbf_var, best_poly_deg, new_poly_var, new_per_var, new_per_period, new_alpha)
                if score < best_score:
                    best_score = score
                    best_ls = new_ls
                    best_rbf_var = new_rbf_var
                    best_poly_var = new_poly_var
                    best_per_var = new_per_var
                    best_per_period = new_per_period
                    best_alpha = new_alpha
            except Exception:
                pass
        
        best_params = (best_ls, best_rbf_var, best_poly_deg, best_poly_var, best_per_var, best_per_period, best_alpha)
        rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha = best_params
    
    # Final fit
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_test_norm = (X - X_mean) / X_std
        preds = ridge_fit_predict(X_norm, y_norm, X_test_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha)
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds
    
    return predict
