"""
Hybrid symbolic regression: combines sparse RBF (k-means centers) with
polynomial and non-linear hand-crafted features, solved via ridge regression.
This avoids the O(n^3) cost of full kernel ridge while retaining locality.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _kmeans(X, k, rng, max_iter=50):
    """Simple k-means clustering."""
    n, d = X.shape
    if k >= n:
        return X.copy()
    # Initialize centers using k-means++ style
    centers = np.empty((k, d))
    centers[0] = X[rng.randint(n)]
    for i in range(1, k):
        d2 = np.min([np.sum((X - centers[j])**2, axis=1) for j in range(i)], axis=0)
        total = d2.sum()
        if total < 1e-12:
            centers[i] = X[rng.randint(n)]
        else:
            probs = d2 / total
            idx = rng.choice(n, p=probs)
            centers[i] = X[idx]
    
    for _ in range(max_iter):
        # Assign points to nearest center
        d2 = np.sum(X**2, axis=1)[:, None] + np.sum(centers**2, axis=1)[None, :] - 2 * X @ centers.T
        assignments = np.argmin(d2, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        
        if np.allclose(centers, new_centers, atol=1e-8):
            centers = new_centers
            break
        centers = new_centers
    
    return centers

def _rbf_features(X, centers, sigma):
    """Compute RBF basis function values: exp(-||x - c_j||^2 / (2*sigma^2))"""
    n = X.shape[0]
    k = centers.shape[0]
    # Vectorized computation
    X2 = np.sum(X**2, axis=1)[:, None]
    C2 = np.sum(centers**2, axis=1)[None, :]
    d2 = X2 + C2 - 2 * X @ centers.T
    np.maximum(d2, 0, out=d2)
    return np.exp(-d2 / (2 * sigma**2))

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 285.0  # safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for stable RBF scaling
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-12] = 1.0
    X_norm = (X - x_min) / x_range
    
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-8
    y_centered = y - y_mean

    # Number of RBF centers: scale with sqrt(n) but cap it
    k_rbf = min(int(np.sqrt(n) * 2), 50)
    
    # Run k-means to find centers
    rng = np.random.RandomState(42)
    centers = _kmeans(X_norm, k_rbf, rng)
    
    # Build hand-crafted features (polynomial + non-linear)
    def build_handcrafted_features(X_norm):
        cols = [np.ones(len(X_norm))]
        # Linear
        for j in range(d):
            cols.append(X_norm[:, j])
        # Quadratic
        for j in range(d):
            cols.append(X_norm[:, j] ** 2)
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X_norm[:, i] * X_norm[:, j])
        # Cubic
        for j in range(d):
            cols.append(X_norm[:, j] ** 3)
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X_norm[:, i] ** 2 * X_norm[:, j])
                cols.append(X_norm[:, i] * X_norm[:, j] ** 2)
        # Non-linear
        for j in range(d):
            xj = X_norm[:, j]
            cols.append(np.sin(np.pi * xj))
            cols.append(np.cos(np.pi * xj))
            cols.append(np.log1p(xj * 5))
            cols.append(1.0 / (1.0 + xj * 5))
        return np.column_stack(cols)
    
    F_handcrafted = build_handcrafted_features(X_norm)
    
    # Initial RBF sigma guess
    # Typical distance scale in [0,1]^d
    sigma_init = 0.3
    
    # Parameters to optimize:
    # log_sigma: log of RBF width
    # log_alpha: log of ridge regularization
    # log_rbf_w: log of RBF weight relative to handcrafted
    # log_hc_w: log of handcrafted weight relative to RBF
    
    # We'll optimize: sigma, alpha, and the relative scaling
    # Actually, let's just optimize sigma and alpha, and fix the relative weights
    # by scaling the features.
    
    # Precompute 5-fold CV
    n_folds = 5
    indices = np.arange(n)
    rng_cv = np.random.RandomState(123)
    rng_cv.shuffle(indices)
    folds = [indices[i::n_folds] for i in range(n_folds)]
    
    def build_full_features(X_norm, sigma):
        F_rbf = _rbf_features(X_norm, centers, sigma)
        F_hc = build_handcrafted_features(X_norm)
        # Scale RBF features to be comparable to handcrafted
        # Divide by sqrt(k) to normalize variance
        F_rbf_scaled = F_rbf / np.sqrt(k_rbf)
        F = np.hstack([F_hc, F_rbf_scaled])
        return F
    
    def objective(x):
        if time.time() > deadline - 10:
            return np.inf
        sigma = np.exp(x[0])
        alpha = np.exp(x[1])
        sigma = np.clip(sigma, 0.01, 5.0)
        alpha = np.clip(alpha, 1e-8, 10.0)
        
        F_train = build_full_features(X_norm, sigma)
        
        cv_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            F_tr = F_train[train_idx]
            y_tr = y_centered[train_idx]
            F_va = F_train[val_idx]
            y_va = y_centered[val_idx]
            
            # Ridge solve
            A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
            try:
                w = np.linalg.solve(A, F_tr.T @ y_tr)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, F_tr.T @ y_tr, rcond=None)[0]
            
            pred = F_va @ w
            mse = np.mean((pred - y_va)**2)
            cv_mses.append(mse)
        
        return np.mean(cv_mses)
    
    # Initial guess
    x0 = np.array([np.log(sigma_init), np.log(1e-4)])
    bounds = [(-3.0, 2.0), (-10.0, 2.0)]
    
    best_x = x0
    best_score = np.inf
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 30, 'ftol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Try a few other starting points if time permits
    if time.time() < deadline - 20:
        for sigma_try in [0.1, 0.2, 0.5, 1.0]:
            x_try = np.array([np.log(sigma_try), np.log(1e-4)])
            try:
                res = minimize(objective, x_try, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 20, 'ftol': 1e-5})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x
            except Exception:
                pass
            if time.time() > deadline - 15:
                break
    
    sigma_opt = np.exp(best_x[0])
    alpha_opt = np.exp(best_x[1])
    sigma_opt = np.clip(sigma_opt, 0.01, 5.0)
    alpha_opt = np.clip(alpha_opt, 1e-8, 10.0)
    
    # Final fit on all data
    F_final = build_full_features(X_norm, sigma_opt)
    A = F_final.T @ F_final + alpha_opt * np.eye(F_final.shape[1])
    try:
        w_final = np.linalg.solve(A, F_final.T @ y_centered)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(A, F_final.T @ y_centered, rcond=None)[0]
    
    def predict(X):
        X_test = np.asarray(X, dtype=np.float64)
        X_test_norm = (X_test - x_min) / x_range
        F_test = build_full_features(X_test_norm, sigma_opt)
        pred = F_test @ w_final + y_mean
        pred = np.nan_to_num(pred, nan=y_mean, posinf=y_mean + 1e6, neginf=y_mean - 1e6)
        return pred
    
    return predict
# EVOLVE-BLOCK-END
