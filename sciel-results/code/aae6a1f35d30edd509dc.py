"""Symbolic regression via adaptive model selection: rich basis, KRR with CV, and GP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 4 equations (300s total)
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_norm = (X - x_mean) / x_std
    
    def time_left():
        return deadline - time.time()
    
    # Split for validation
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(0.8 * n)
    idx_train = indices[:split]
    idx_val = indices[split:]
    
    X_tr = X[idx_train]
    y_tr = y[idx_train]
    X_val = X[idx_val]
    y_val = y[idx_val]
    
    X_tr_norm = X_norm[idx_train]
    X_val_norm = X_norm[idx_val]
    
    best_val_mse = np.inf
    best_predict = None
    best_params = {}
    
    # --- Helper: build rich feature matrix ---
    def build_features(X_in):
        x = X_in[:, 0]
        cols = []
        # Polynomials up to degree 10
        for i in range(11):
            cols.append(x**i)
        # Trigonometric
        for k in [1, 2, 3, 4, 5, 6, 7]:
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        # Exponentials
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.exp(0.5 * x))
        cols.append(np.exp(-0.5 * x))
        # Other non-linear
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log1p(np.abs(x)))
        cols.append(np.sign(x) * np.sqrt(np.abs(x)))
        cols.append(np.sqrt(np.abs(x)) * np.sign(x))
        # Rational-like
        cols.append(x / (1 + x**2))
        cols.append(x / (1 + x**4))
        # Spline-like: piecewise linear bumps
        for center in np.linspace(-3, 3, 9):
            cols.append(np.abs(x - center))
            cols.append(np.exp(-((x - center)**2) / 2.0))
            cols.append(np.exp(-((x - center)**2) / 0.5))
        return np.column_stack(cols)
    
    F_tr = build_features(X_tr)
    F_val = build_features(X_val)
    
    # --- Model 1: Ridge Regression with rich features ---
    alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    for alpha in alphas:
        if time_left() < 10:
            break
        try:
            A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
            b = F_tr.T @ y_tr
            w = np.linalg.solve(A, b)
            
            y_pred_val = F_val @ w
            mse = np.mean((y_pred_val - y_val)**2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_predict = 'linear'
                best_params = {'w': w, 'alpha': alpha}
        except np.linalg.LinAlgError:
            pass
    
    # --- Model 2: Kernel Ridge Regression (RBF) with proper CV ---
    def rbf_kernel(X1, X2, gamma):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-gamma * sq_dists)
    
    # Use 5-fold CV for KRR
    n_folds = 5
    indices_cv = np.arange(n)
    rng_cv = np.random.RandomState(123)
    rng_cv.shuffle(indices_cv)
    folds = np.array_split(indices_cv, n_folds)
    
    gamma_grid = np.logspace(-3, 4, 30)
    lam_grid = np.logspace(-5, 3, 15)
    
    best_krr_mse = np.inf
    best_krr_gamma = 1.0
    best_krr_lam = 1e-3
    
    for gamma in gamma_grid:
        if time_left() < 15:
            break
            
        K_full = rbf_kernel(X_norm, X_norm, gamma)
        
        for lam in lam_grid:
            if time_left() < 10:
                    break
            cv_errors = []
            for i in range(n_folds):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                K_tr = K_full[np.ix_(tr_idx, tr_idx)]
                K_val = K_full[np.ix_(val_idx, tr_idx)]
                
                K_reg = K_tr + lam * np.eye(len(tr_idx))
                
                try:
                    w = np.linalg.solve(K_reg, y[tr_idx])
                    pred = K_val @ w
                    cv_errors.append(np.mean((pred - y[val_idx])**2))
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
            
            mean_cv_mse = np.mean(cv_errors)
            
            if mean_cv_mse < best_krr_mse:
                best_krr_mse = mean_cv_mse
                best_krr_gamma = gamma
                best_krr_lam = lam
            
            if mean_cv_mse < best_val_mse:
                best_val_mse = mean_cv_mse
                best_predict = 'krr'
                best_params = {'gamma': gamma, 'lam': lam}
    
    # --- Model 3: Gaussian Process Regression ---
    length_scales = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    noise_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    best_gp_mse = np.inf
    best_gp_ls = 1.0
    best_gp_noise = 1e-4
    
    for ls in length_scales:
        if time_left() < 15:
            break
        gamma_gp = 1.0 / (2 * ls**2)
        K_tr_gp = rbf_kernel(X_tr_norm, X_tr_norm, gamma_gp)
        for noise in noise_levels:
            if time_left() < 10:
                break
            try:
                A = K_tr_gp + noise * np.eye(len(X_tr_norm))
                L = np.linalg.cholesky(A)
                alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                
                K_val_gp = rbf_kernel(X_val_norm, X_tr_norm, gamma_gp)
                y_pred_val = K_val_gp @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_gp_mse:
                    best_gp_mse = mse
                    best_gp_ls = ls
                    best_gp_noise = noise
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_predict = 'gp'
                    best_params = {'ls': ls, 'noise': noise}
            except np.linalg.LinAlgError:
                pass
    
    # Refit on all data with best model
    if best_predict == 'linear' and 'w' in best_params:
        F_all = build_features(X)
        alpha = best_params['alpha']
        A = F_all.T @ F_all + alpha * np.eye(F_all.shape[1])
        b = F_all.T @ y
        try:
            w_final = np.linalg.solve(A, b)
            
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                F_pred = build_features(X_pred)
                return F_pred @ w_final
        except:
            def predict(X_pred):
                return np.zeros(len(np.asarray(X_pred)))
        
    elif best_predict == 'krr' and 'gamma' in best_params:
        gamma = best_params['gamma']
        lam = best_params['lam']
        K_all = rbf_kernel(X_norm, X_norm, gamma)
        A = K_all + lam * np.eye(n)
        try:
            alpha_final = np.linalg.solve(A, y)
            
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                X_pred_norm = (X_pred - x_mean) / x_std
                K_pred = rbf_kernel(X_pred_norm, X_norm, gamma)
                pred = K_pred @ alpha_final
                return np.where(np.isfinite(pred), pred, 0.0)
        except:
            def predict(X_pred):
                return np.zeros(len(np.asarray(X_pred)))
    
    elif best_predict == 'gp' and 'ls' in best_params:
        gamma_gp = 1.0 / (2 * best_params['ls']**2)
        noise = best_params['noise']
        K_all = rbf_kernel(X_norm, X_norm, gamma_gp)
        A = K_all + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(A)
            alpha_final = np.linalg.solve(L.T, np.linalg.solve(L, y))
            
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                X_pred_norm = (X_pred - x_mean) / x_std
                K_pred = rbf_kernel(X_pred_norm, X_norm, gamma_gp)
                pred = K_pred @ alpha_final
                return np.where(np.isfinite(pred), pred, 0.0)
        except:
            def predict(X_pred):
                return np.zeros(len(np.asarray(X_pred)))
    
    else:
        # Fallback: polynomial fit
        try:
            x = X[:, 0]
            coeffs = np.polyfit(x, y, 6)
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                return np.polyval(coeffs, X_pred[:, 0])
        except:
            def predict(X_pred):
                return np.zeros(len(np.asarray(X_pred)))
    
    return predict

# EVOLVE-BLOCK-END
