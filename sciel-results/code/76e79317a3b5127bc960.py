"""Symbolic regression via adaptive model selection with spline interpolation for 1D noiseless data."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 270  # Safety margin for 4 equations (300s total)
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    d = X.shape[1]
    
    def time_left():
        return deadline - time.time()
    
    # --- Strategy 1: Cubic spline interpolation (best for smooth 1D noiseless data) ---
    # Sort by x
    if d == 1:
        x = X[:, 0]
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Check if data is reasonably smooth for spline
        # Compute second differences to check for smoothness
        if n >= 3:
            dx = np.diff(x_sorted)
            # Check for duplicates
            if np.all(dx > 1e-10):
                # Try cubic spline
                try:
                    cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')
                    
                    # Validate on a subset
                    val_idx = np.random.RandomState(42).choice(n, max(20, n//10), replace=False)
                    val_x = x[val_idx]
                    val_y = y[val_idx]
                    val_pred = cs(val_x)
                    val_mse = np.mean((val_pred - val_y)**2)
                    y_var = np.var(y)
                    if y_var > 0:
                        val_nmse = val_mse / y_var
                    else:
                        val_nmse = val_mse
                    
                    # Spline is excellent if validation NMSE is very small
                    if val_nmse < 1e-6:
                        def predict(X_pred):
                            X_pred = np.asarray(X_pred, dtype=np.float64)
                            if X_pred.ndim == 1:
                                X_pred = X_pred.reshape(-1, 1)
                            xp = X_pred[:, 0]
                            # Clip to training range to avoid extrapolation issues
                            xp_clipped = np.clip(xp, x_sorted[0], x_sorted[-1])
                            return cs(xp_clipped)
                        return predict
                except:
                    pass
        
        # If spline didn't work well, try linear interpolation as fallback for 1D
        if np.all(np.diff(x_sorted) > 1e-10):
            try:
                f_interp = interp1d(x_sorted, y_sorted, kind='cubic', fill_value='extrapolate')
                
                # Validate
                val_idx = np.random.RandomState(42).choice(n, max(20, n//10), replace=False)
                val_x = x[val_idx]
                val_y = y[val_idx]
                val_pred = f_interp(val_x)
                val_mse = np.mean((val_pred - val_y)**2)
                y_var = np.var(y)
                if y_var > 0:
                    val_nmse = val_mse / y_var
                else:
                    val_nmse = val_mse
                
                if val_nmse < 1e-4:
                    def predict(X_pred):
                        X_pred = np.asarray(X_pred, dtype=np.float64)
                        if X_pred.ndim == 1:
                            X_pred = X_pred.reshape(-1, 1)
                        return f_interp(X_pred[:, 0])
                    return predict
            except:
                pass
    
    # --- Strategy 2: Kernel Ridge Regression with adaptive hyperparameters ---
    # Use all data for final fit, but select hyperparams via simple heuristic
    
    x = X[:, 0] if d == 1 else X[:, 0]
    
    # Normalize x for kernel
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0
    
    def rbf_kernel(X1, X2, gamma):
        """RBF kernel with precomputed distances."""
        # X1, X2 are 1D arrays of x values
        sq_dists = (X1[:, None] - X2[None, :])**2
        return np.exp(-gamma * sq_dists)
    
    # Try a range of gamma values
    # Optimal gamma depends on spacing of points
    dx_mean = np.mean(np.diff(np.sort(x)))
    if dx_mean > 0:
        # gamma ~ 1/(2*dx^2) is a reasonable starting point
        gamma_base = 1.0 / (2 * dx_mean**2)
    else:
        gamma_base = 1.0
    
    gammas_to_try = [gamma_base * f for f in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]]
    
    # Also try some absolute values
    gammas_to_try.extend([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    gammas_to_try = sorted(set(gammas_to_try))
    
    # Keep unique and reasonable
    gammas_to_try = [g for g in gammas_to_try if 1e-6 < g < 1e6]
    
    best_val_mse = np.inf
    best_gamma = 1.0
    best_alpha = 1e-4
    best_predict_type = None
    
    # Use 80/20 split for validation
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(0.8 * n)
    idx_train = indices[:split]
    idx_val = indices[split:]
    
    x_tr = x[idx_train]
    y_tr = y[idx_train]
    x_val = x[idx_val]
    y_val = y[idx_val]
    
    K_tr = rbf_kernel(x_tr, x_tr, 1.0)  # placeholder, will recompute
    
    # Try different regularization strengths
    lambdas = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    for gamma in gammas_to_try:
        if time_left() < 10:
            break
        K_tr = rbf_kernel(x_tr, x_tr, gamma)
        for lam in lambdas:
            if time_left() < 5:
                break
            try:
                A = K_tr + lam * np.eye(len(x_tr))
                alpha_vec = np.linalg.solve(A, y_tr)
                
                K_val = rbf_kernel(x_val, x_tr, gamma)
                y_pred_val = K_val @ alpha_vec
                mse = np.mean((y_pred_val - y_val)**2)
                
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_gamma = gamma
                    best_alpha = lam
                    best_predict_type = 'krr'
            except np.linalg.LinAlgError:
                pass
    
    # --- Strategy 3: Rich linear basis with ridge regression ---
    def build_features(X_in):
        x = X_in[:, 0] if X_in.shape[1] == 1 else X_in[:, 0]
        cols = []
        # Polynomials up to degree 10
        for i in range(11):
            cols.append(x**i)
        # Trigonometric
        for k in [1, 2, 3, 4, 5, 6, 8, 10]:
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        # Exponentials
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.exp(0.5 * x))
        cols.append(np.exp(-0.5 * x))
        # Other non-linear
        cols.append(np.tanh(x))
        cols.append(np.tanh(2 * x))
        cols.append(np.arctan(x))
        cols.append(np.log1p(np.abs(x)))
        cols.append(np.sign(x) * np.sqrt(np.abs(x)))
        cols.append(np.sign(x) * np.abs(x)**0.75)
        cols.append(np.sign(x) * np.abs(x)**1.5)
        # Spline-like bumps
        for center in np.linspace(-4, 4, 9):
            cols.append(np.abs(x - center))
            cols.append(np.exp(-((x - center)**2) / 1.0))
            cols.append(np.exp(-((x - center)**2) / 4.0))
        return np.column_stack(cols)
    
    F_tr = build_features(X[idx_train])
    F_val = build_features(X[idx_val])
    
    alphas = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    for alpha in alphas:
        if time_left() < 5:
            break
        try:
            A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
            b = F_tr.T @ y_tr
            w = np.linalg.solve(A, b)
            
            y_pred_val = F_val @ w
            mse = np.mean((y_pred_val - y_val)**2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_gamma = None  # marker for linear
                best_alpha = alpha
                best_predict_type = 'linear'
        except np.linalg.LinAlgError:
            pass
    
    # Refit on all data with best model
    if best_predict_type == 'krr':
        K_all = rbf_kernel(x, x, best_gamma)
        A = K_all + best_alpha * np.eye(n)
        try:
            alpha_final = np.linalg.solve(A, y)
            
            def predict(X_pred):
                X_pred = np.asarray(X_pred, dtype=np.float64)
                if X_pred.ndim == 1:
                    X_pred = X_pred.reshape(-1, 1)
                xp = X_pred[:, 0]
                K_pred = rbf_kernel(xp, x, best_gamma)
                return K_pred @ alpha_final
        except:
            # Fallback to simple interpolation
            try:
                sort_idx = np.argsort(x)
                f_interp = interp1d(x[sort_idx], y[sort_idx], kind='linear', fill_value='extrapolate')
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    return f_interp(X_pred[:, 0])
            except:
                def predict(X_pred):
                    return np.zeros(len(np.asarray(X_pred)))
    
    elif best_predict_type == 'linear':
        F_all = build_features(X)
        A = F_all.T @ F_all + best_alpha * np.eye(F_all.shape[1])
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
            try:
                sort_idx = np.argsort(x)
                f_interp = interp1d(x[sort_idx], y[sort_idx], kind='linear', fill_value='extrapolate')
                def predict(X_pred):
                    X_pred = np.asarray(X_pred, dtype=np.float64)
                    if X_pred.ndim == 1:
                        X_pred = X_pred.reshape(-1, 1)
                    return f_interp(X_pred[:, 0])
            except:
                def predict(X_pred):
                    return np.zeros(len(np.asarray(X_pred)))
    
    else:
        # Fallback: polynomial fit
        try:
            coeffs = np.polyfit(x, y, min(8, n-1))
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
