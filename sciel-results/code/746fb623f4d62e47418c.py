# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    def check_time():
        return time.time() < deadline
    
    # Normalize features for numerical stability
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    # Normalize y
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf
    
    best_predict = None
    best_mse = np.inf
    
    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_mse:
            best_mse = mse
            best_predict = model
    
    def ridge_solve(F, target, lam=1e-5):
        A = F.T @ F + lam * np.eye(F.shape[1])
        A[0, 0] -= lam  # Don't regularize intercept
        return solve(A, F.T @ target)
    
    # Strategy 1: Sum-of-Sines kernel with dense frequency grid
    # This uses a fixed set of sinusoidal basis functions with varying frequencies
    # to capture smooth periodic and non-periodic patterns
    def make_sines_features(X, max_freq, n_harmonics_per_dim):
        Xn = (X - X_min) / X_range
        cols = [np.ones(len(Xn))]
        
        # For each dimension, add sin and cos at multiple frequencies
        for j in range(Xn.shape[1]):
            x = Xn[:, j]
            for k in range(1, max_freq + 1):
                freq = k * np.pi
                cols.append(np.sin(freq * x))
                cols.append(np.cos(freq * x))
        
        # Cross terms: sin/cos of sums and differences of normalized dims
        if Xn.shape[1] >= 2:
            for i in range(Xn.shape[1]):
                for j in range(i + 1, Xn.shape[1]):
                    for k in range(1, min(max_freq, 3) + 1):
                        freq = k * np.pi
                        # Sum
                        cols.append(np.sin(freq * (Xn[:, i] + Xn[:, j])))
                        cols.append(np.cos(freq * (Xn[:, i] + Xn[:, j])))
                        # Difference
                        cols.append(np.sin(freq * (Xn[:, i] - Xn[:, j])))
                        cols.append(np.cos(freq * (Xn[:, i] - Xn[:, j])))
                        # Product
                        cols.append(np.sin(freq * Xn[:, i] * Xn[:, j]))
                        cols.append(np.cos(freq * Xn[:, i] * Xn[:, j]))
        
        return np.column_stack(cols)
    
    # Try different frequency ranges and regularization strengths
    for max_freq in [3, 5, 8, 12]:
        if not check_time():
            break
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            if not check_time():
                break
            try:
                F = make_sines_features(X_train, max_freq, d)
                w = ridge_solve(F, y_norm, lam)
                
                def predict_sines(X, w, y_mean, y_std, mf):
                    F_test = make_sines_features(X, mf, d)
                    return y_mean + y_std * (F_test @ w)
                
                update_best(lambda X, w=w, mf=max_freq: predict_sines(X, w, y_mean, y_std, mf))
            except Exception:
                pass
    
    # Strategy 2: Chebyshev polynomial basis (better conditioned)
    def make_cheb_features(X, degree):
        Xn = (X - X_min) / X_range
        cols = [np.ones(len(Xn))]
        for j in range(Xn.shape[1]):
            # Chebyshev polynomials of first kind
            x = 2 * Xn[:, j] - 1  # map to [-1, 1]
            T0 = np.ones(len(x))
            T1 = x.copy()
            cols.append(T1)
            for k in range(2, degree + 1):
                Tk = 2 * x * T1 - T0
                cols.append(Tk)
                T0, T1 = T1, Tk
        # Interaction terms (up to degree 2 for simplicity)
        if Xn.shape[1] >= 2:
            for i in range(Xn.shape[1]):
                for j in range(i + 1, Xn.shape[1]):
                    cols.append(Xn[:, i] * Xn[:, j])
                    cols.append(np.sin(np.pi * Xn[:, i] * Xn[:, j]))
        return np.column_stack(cols)
    
    for deg in [2, 3, 4, 5]:
        if not check_time():
            break
        try:
            F = make_cheb_features(X_train, deg)
            w = ridge_solve(F, y_norm)
            
            def predict_cheb(X, w, y_mean, y_std):
                F_test = make_cheb_features(X, deg)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X, w=w, d=deg: predict_cheb(X, w, y_mean, y_std))
        except Exception:
            pass
    
    # Strategy 3: RBF with multiple center selections and gammas
    def make_rbf_model(gamma, n_centers, seed=42):
        np.random.seed(seed)
        if n > n_centers:
            idx = np.random.choice(n, n_centers, replace=False)
            centers = X_train[idx]
        else:
            centers = X_train
        
        diffs = X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F = np.exp(-gamma * dists)
        
        w = ridge_solve(F, y_norm)
        
        def predict_rbf(X, centers, gamma, w, y_mean, y_std):
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            F_test = np.exp(-gamma * dists)
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_rbf(X, centers, gamma, w, y_mean, y_std)
    
    # Estimate scale for gamma
    X_range_mean = np.mean(X_range)
    base_gamma = 1.0 / (d * X_range_mean ** 2 + 1e-10)
    
    for g_mult in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        if not check_time():
            break
        for nc in [30, 50]:
            try:
                model = make_rbf_model(base_gamma * g_mult, nc)
                update_best(model)
            except Exception:
                pass
    
    # Strategy 4: ReLU-based model
    def make_relu_model(n_hidden, alpha, seed=42):
        np.random.seed(seed)
        W1 = np.random.randn(d, n_hidden) * alpha
        b1 = np.zeros(n_hidden)
        
        def forward(Xn):
            z1 = Xn @ W1 + b1
            a1 = np.maximum(0, z1)
            return a1
        
        F = forward(X_norm)
        F = np.column_stack([np.ones(len(F)), F])
        
        w = ridge_solve(F, y_norm)
        
        def predict_relu(X, W1, b1, w, y_mean, y_std):
            Xn = (X - X_min) / X_range
            z1 = Xn @ W1 + b1
            a1 = np.maximum(0, z1)
            F_test = np.column_stack([np.ones(len(a1)), a1])
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_relu(X, W1, b1, w, y_mean, y_std)
    
    for n_hid in [15, 30]:
        if not check_time():
            break
        try:
            model = make_relu_model(n_hid, 1.0)
            update_best(model)
        except Exception:
            pass
    
    # Strategy 5: Simple functional forms (exp, log, sqrt, sin, cos)
    def make_special_features(X):
        Xn = (X - X_min) / X_range
        cols = [np.ones(len(Xn))]
        for j in range(Xn.shape[1]):
            x = Xn[:, j]
            cols.append(x)
            cols.append(np.exp(-x))
            cols.append(np.exp(x - 1))
            cols.append(np.sqrt(np.maximum(x, 1e-10)))
            cols.append(np.log(x + 1e-10))
            cols.append(np.sin(np.pi * x))
            cols.append(np.cos(np.pi * x))
            cols.append(np.sin(2 * np.pi * x))
            cols.append(np.cos(2 * np.pi * x))
            cols.append(1.0 / (x + 0.5))
            cols.append(x * np.sin(np.pi * x))
            cols.append(x * np.exp(-x))
        # Pairwise
        if Xn.shape[1] >= 2:
            for i in range(Xn.shape[1]):
                for j in range(i + 1, Xn.shape[1]):
                    cols.append(Xn[:, i] * Xn[:, j])
                    cols.append(np.exp(-Xn[:, i] * Xn[:, j]))
                    cols.append(np.sin(np.pi * Xn[:, i] * Xn[:, j]))
        return np.column_stack(cols)
    
    if check_time():
        try:
            F = make_special_features(X_train)
            w = ridge_solve(F, y_norm)
            
            def predict_special(X, w, y_mean, y_std):
                F_test = make_special_features(X)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X, w=w: predict_special(X, w, y_mean, y_std))
        except Exception:
            pass
    
    # Strategy 6: 1D fits for each dimension
    for i in range(d):
        if not check_time():
            break
        try:
            x1 = X_train[:, i]
            for deg in [3, 5]:
                coeffs = np.polyfit(x1, y_norm, deg)
                def predict_1d(X, col, coeffs, y_mean, y_std):
                    return y_mean + y_std * np.polyval(coeffs, X[:, col])
                update_best(lambda X, c=i, co=coeffs: predict_1d(X, c, co, y_mean, y_std))
        except Exception:
            pass
    
    # Strategy 7: Pairwise polynomial fits
    for i in range(d):
        for j in range(i + 1, d):
            if not check_time():
                break
            try:
                x1 = X_train[:, i]
                x2 = X_train[:, j]
                F = np.column_stack([
                    np.ones(n),
                    x1, x2,
                    x1 * x2,
                    x1 ** 2, x2 ** 2,
                    x1 ** 3, x2 ** 3,
                    x1 ** 2 * x2, x1 * x2 ** 2,
                    x1 ** 3 * x2, x1 * x2 ** 3,
                    x1 ** 2 * x2 ** 2,
                    np.sin(np.pi * x1 / X_range[i]),
                    np.cos(np.pi * x1 / X_range[i]),
                    np.sin(np.pi * x2 / X_range[j]),
                    np.cos(np.pi * x2 / X_range[j]),
                ])
                w = ridge_solve(F, y_norm)
                
                def predict_pair(X, i, j, w, y_mean, y_std):
                    x1 = X[:, i]
                    x2 = X[:, j]
                    F_test = np.column_stack([
                        np.ones(len(x1)),
                        x1, x2,
                        x1 * x2,
                        x1 ** 2, x2 ** 2,
                        x1 ** 3, x2 ** 3,
                        x1 ** 2 * x2, x1 * x2 ** 2,
                        x1 ** 3 * x2, x1 * x2 ** 3,
                        x1 ** 2 * x2 ** 2,
                        np.sin(np.pi * x1 / X_range[i]),
                        np.cos(np.pi * x1 / X_range[i]),
                        np.sin(np.pi * x2 / X_range[j]),
                        np.cos(np.pi * x2 / X_range[j]),
                    ])
                    return y_mean + y_std * (F_test @ w)
                
                update_best(lambda X, i=i, j=j, w=w: predict_pair(X, i, j, w, y_mean, y_std))
            except Exception:
                pass
    
    # Fallback
    if best_predict is None:
        return lambda X: np.full(len(X), y_mean)
    
    return best_predict

# EVOLVE-BLOCK-END
