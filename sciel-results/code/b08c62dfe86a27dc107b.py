"""Symbolic regression via adaptive model selection between rich linear basis and Kernel Ridge Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 270  # Safety margin for 4 equations (300s total)
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    d = X.shape[1]
    
    # If X is 1D, make it 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
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
    
    best_val_mse = np.inf
    best_predict = None
    
    # --- Helper: build rich feature matrix ---
    def build_features(X_in):
        x = X_in[:, 0] if X_in.shape[1] == 1 else X_in[:, 0]
        cols = []
        # Polynomials up to degree 8
        for i in range(9):
            cols.append(x**i)
        # Trigonometric
        for k in [1, 2, 3, 4, 5]:
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        # Exponentials
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        # Other non-linear
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log1p(np.abs(x)))
        cols.append(np.sign(x) * np.sqrt(np.abs(x)))
        # Spline-like: piecewise linear bumps
        for center in np.linspace(-3, 3, 7):
            cols.append(np.abs(x - center))
            cols.append(np.exp(-((x - center)**2) / 2.0))
        return np.column_stack(cols)
    
    F_tr = build_features(X_tr)
    F_val = build_features(X_val)
    
    # --- Model 1: Ridge Regression with rich features ---
    alphas = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best_linear_params = None
    best_linear_alpha = 1e-4
    
    for alpha in alphas:
        try:
            A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
            b = F_tr.T @ y_tr
            w = np.linalg.solve(A, b)
            
            y_pred_val = F_val @ w
            mse = np.mean((y_pred_val - y_val)**2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_linear_params = w
                best_linear_alpha = alpha
                best_predict = 'linear'
        except np.linalg.LinAlgError:
            pass
    
    # --- Model 2: Kernel Ridge Regression (RBF) ---
    def rbf_kernel(X1, X2, gamma):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        np.maximum(sq_dists, 0, out=sq_dists)
        return np.exp(-gamma * sq_dists)
    
    gammas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    best_krr_params = None
    best_krr_gamma = 1.0
    best_krr_lambda = 1e-3
    
    for gamma in gammas:
        K_tr = rbf_kernel(X_tr, X_tr, gamma)
        for lam in lambdas:
            try:
                A = K_tr + lam * np.eye(len(X_tr))
                alpha_vec = np.linalg.solve(A, y_tr)
                
                K_val = rbf_kernel(X_val, X_tr, gamma)
                y_pred_val = K_val @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_krr_params = alpha_vec
                    best_krr_gamma = gamma
                    best_krr_lambda = lam
                    best_predict = 'krr'
            except np.linalg.LinAlgError:
                pass
    
    # --- Model 3: Gaussian Process Regression (simple) ---
    # Try a few length scales
    length_scales = [0.5, 1.0, 2.0, 5.0]
    noise_levels = [1e-6, 1e-5, 1e-4, 1e-3]
    
    best_gp_params = None
    best_gp_ls = 1.0
    best_gp_noise = 1e-4
    
    for ls in length_scales:
        gamma_gp = 1.0 / (2 * ls**2)
        K_tr_gp = rbf_kernel(X_tr, X_tr, gamma_gp)
        for noise in noise_levels:
            try:
                A = K_tr_gp + noise * np.eye(len(X_tr))
                L = np.linalg.cholesky(A)
                alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                
                K_val_gp = rbf_kernel(X_val, X_tr, gamma_gp)
                y_pred_val = K_val_gp @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_gp_params = alpha_vec
                    best_gp_ls = ls
                    best_gp_noise = noise
                    best_predict = 'gp'
            except np.linalg.LinAlgError:
                pass
    
    # Check time
    if time.time() > deadline - 30:
        # Use whatever we have so far
        pass
    
    # Refit on all data with best model
    if best_predict == 'linear' and best_linear_params is not None:
        F_all = build_features(X)
        A = F_all.T @ F_all + best_linear_alpha * np.eye(F_all.shape[1])
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
            # Fallback
            def predict(X_pred):
                return np.zeros(len(np.asarray(X_pred)))
        
    elif best_predict == 'krr' and best_krr_params is not None:
        gamma = best_krr_gamma
        lam = best_krr_lambda
        K_all = rbf_kernel(X, X, gamma)
        A = K_all + lam * np.eye(len(X))
        try:
            alpha_final = np.linalg.solve(A, y)
            
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                K_pred = rbf_kernel(X_pred, X, gamma)
                return K_pred @ alpha_final
        except:
            def predict(X_pred):
                return np.zeros(len(np.asarray(X_pred)))
    
    elif best_predict == 'gp' and best_gp_params is not None:
        gamma_gp = 1.0 / (2 * best_gp_ls**2)
        noise = best_gp_noise
        K_all = rbf_kernel(X, X, gamma_gp)
        A = K_all + noise * np.eye(len(X))
        try:
            L = np.linalg.cholesky(A)
            alpha_final = np.linalg.solve(L.T, np.linalg.solve(L, y))
            
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                K_pred = rbf_kernel(X_pred, X, gamma_gp)
                return K_pred @ alpha_final
        except:
            def predict(X_pred):
                return np.zeros(len(np.asarray(X_pred)))
    
    else:
        # Fallback: simple polynomial fit
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
