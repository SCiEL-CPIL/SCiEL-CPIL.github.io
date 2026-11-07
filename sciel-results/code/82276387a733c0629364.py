"""Symbolic regression via multi-model search: polynomial, spline, GP, and NN."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, Rbf
from scipy.optimize import minimize
from scipy.signal import savgol_filter

def _relu(x):
    return np.maximum(0, x)

def _fit_polynomial(X_train, y_train, degree):
    """Fit a polynomial of given degree."""
    d = X_train.shape[1]
    if d != 1:
        return None
    
    x = X_train[:, 0]
    # Design matrix
    X_design = np.vander(x, N=degree+1, increasing=True)
    
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_design, y_train, rcond=None)
    except Exception:
        return None
    
    def predict(X):
        x_test = X[:, 0]
        X_design_test = np.vander(x_test, N=degree+1, increasing=True)
        return X_design_test @ coeffs
    
    return predict

def _fit_spline(X_train, y_train, method='pchip'):
    """Fit a spline interpolator."""
    d = X_train.shape[1]
    if d != 1:
        return None
    
    x = X_train[:, 0]
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y_train[sort_idx]
    
    if method == 'pchip':
        spl = PchipInterpolator(x_sorted, y_sorted)
    else:
        spl = CubicSpline(x_sorted, y_sorted)
        
    def predict(X):
        return spl(X[:, 0])
    
    return predict

def _fit_rbf(X_train, y_train, kernel='gaussian', epsilon=1.0):
    """Fit a Radial Basis Function interpolator."""
    d = X_train.shape[1]
    if d != 1:
        return None
    
    x = X_train[:, 0]
    try:
        rbf = Rbf(x, y_train, function=kernel, epsilon=epsilon)
    except Exception:
        return None
    
    def predict(X):
        return rbf(X[:, 0])
    
    return predict

def _fit_gaussian_process(X_train, y_train, deadline=None):
    """Fit a Gaussian Process with optimized hyperparameters."""
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    Xs = (X_train - X_mean) / X_std
    
    def kernel(X1, X2, log_l, log_sf):
        l = np.exp(log_l)
        sf = np.exp(log_sf)
        sq_dist = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=2)
        return (sf**2) * np.exp(-0.5 * sq_dist / (l**2))
    
    def nlml(params):
        log_l, log_sf = params
        sigma_n = 1e-6
        K = kernel(Xs, Xs, log_l, log_sf) + sigma_n * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            nlml_val = 0.5 * y_train @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            return nlml_val
        except np.linalg.LinAlgError:
            return 1e10

    best_nll = float('inf')
    best_params = [0.0, 0.0]
    
    starts = [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [-0.5, 0.5],
        [0.5, -0.5]
    ]
    
    for start in starts:
        if deadline is not None and time.time() > deadline - 3:
            break
        try:
            res = minimize(nlml, start, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            continue
    
    log_l, log_sf = best_params
    sigma_n = 1e-6
    
    K = kernel(Xs, Xs, log_l, log_sf) + sigma_n * np.eye(n)
    K_inv_y = np.linalg.solve(K, y_train)
    
    def predict(X):
        Xs_test = (X - X_mean) / X_std
        K_star = kernel(Xs_test, Xs, log_l, log_sf)
        return K_star @ K_inv_y
        
    return predict

def _fit_nn(X_train, y_train, hidden_sizes, lr, epochs, seed=0, deadline=None):
    """Train a small MLP with Adam optimizer."""
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]
    n = X_train.shape[0]
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    layers = [d] + list(hidden_sizes) + [1]
    L = len(layers)
    
    W = [rng.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(L-1)]
    b = [np.zeros(layers[i+1]) for i in range(L-1)]
    
    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(bi) for bi in b]
    vb = [np.zeros_like(bi) for bi in b]
    t = 0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    best_loss = float('inf')
    best_W = [w.copy() for w in W]
    best_b = [bi.copy() for bi in b]
    
    for ep in range(epochs):
        if deadline is not None and time.time() > deadline:
            break
        
        # Forward
        a = X_train
        cache = [a]
        for i in range(L-1):
            z = a @ W[i] + b[i]
            a = _relu(z) if i < L-2 else z
            cache.append(z)
            cache.append(a)
        
        loss = np.mean((a - y_norm)**2)
        if loss < best_loss:
            best_loss = loss
            best_W = [w.copy() for w in W]
            best_b = [bi.copy() for bi in b]
            
        # Backward
        da = 2.0 * (a - y_norm) / n
        for i in range(L-2, -1, -1):
            dz = da.copy()
            if i < L-2:
                dz *= (cache[2*(i+1)] > 0)
            dW = cache[2*i+1].T @ dz
            db = dz.sum(axis=0)
            
            t += 1
            mW[i] = beta1 * mW[i] + (1 - beta1) * dW
            vW[i] = beta2 * vW[i] + (1 - beta2) * (dW**2)
            mW_hat = mW[i] / (1 - beta1**t)
            vW_hat = vW[i] / (1 - beta2**t)
            W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            mb[i] = beta1 * mb[i] + (1 - beta1) * db
            vb[i] = beta2 * vb[i] + (1 - beta2) * (db**2)
            mb_hat = mb[i] / (1 - beta1**t)
            vb_hat = vb[i] / (1 - beta2**t)
            b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
            
    def predict(X):
        a = X
        for i in range(L-2):
            a = _relu(a @ best_W[i] + best_b[i])
        a = a @ best_W[L-2] + best_b[L-2]
        return a * y_std + y_mean
    
    return predict, best_loss

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 270  # Safety margin for 300s limit
    
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    candidates = []
    
    # 1. Polynomial fits (fast, good for smooth functions)
    if d == 1:
        for deg in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
            if time.time() > deadline - 30:
                break
            try:
                poly_pred = _fit_polynomial(X_train, y_train, degree=deg)
                if poly_pred:
                    y_pred = poly_pred(X_train)
                    mse = np.mean((y_pred - y_train)**2)
                    candidates.append((mse, poly_pred, 'poly'))
            except Exception:
                pass
        
        # 2. Spline fits
        try:
            spl_pred = _fit_spline(X_train, y_train, method='pchip')
            if spl_pred:
                y_pred = spl_pred(X_train)
                mse = np.mean((y_pred - y_train)**2)
                candidates.append((mse, spl_pred, 'spline'))
        except Exception:
            pass
        
        try:
            spl_pred = _fit_spline(X_train, y_train, method='cubic')
            if spl_pred:
                y_pred = spl_pred(X_train)
                mse = np.mean((y_pred - y_train)**2)
                candidates.append((mse, spl_pred, 'spline'))
        except Exception:
            pass
        
        # 3. RBF fits
        for eps in [0.5, 1.0, 2.0, 5.0]:
            if time.time() > deadline - 20:
                break
            try:
                rbf_pred = _fit_rbf(X_train, y_train, kernel='gaussian', epsilon=eps)
                if rbf_pred:
                    y_pred = rbf_pred(X_train)
                    mse = np.mean((y_pred - y_train)**2)
                    candidates.append((mse, rbf_pred, 'rbf'))
            except Exception:
                pass

    # 4. Gaussian Process
    if time.time() < deadline - 20:
        try:
            gp_pred = _fit_gaussian_process(X_train, y_train, deadline=deadline - 10)
            if gp_pred:
                y_pred = gp_pred(X_train)
                mse = np.mean((y_pred - y_train)**2)
                candidates.append((mse, gp_pred, 'gp'))
        except Exception:
            pass
        
    # 5. Neural Networks
    nn_configs = [
        {"hidden_sizes": [32, 32], "lr": 0.01, "epochs": 1500, "seed": 0},
        {"hidden_sizes": [64, 32], "lr": 0.005, "epochs": 1500, "seed": 1},
        {"hidden_sizes": [16, 16, 16], "lr": 0.005, "epochs": 1000, "seed": 2},
    ]
    
    for cfg in nn_configs:
        if time.time() > deadline - 15:
            break
        try:
            nn_pred, loss = _fit_nn(X_train, y_train, 
                                    hidden_sizes=cfg["hidden_sizes"],
                                    lr=cfg["lr"],
                                    epochs=cfg["epochs"],
                                    seed=cfg["seed"],
                                    deadline=deadline - 5)
            y_pred = nn_pred(X_train)
            mse = np.mean((y_pred - y_train)**2)
            candidates.append((mse, nn_pred, 'nn'))
        except Exception:
            continue
            
    if not candidates:
        mean_val = np.mean(y_train)
        def predict(X):
            return np.full(X.shape[0], mean_val)
        return predict
        
    # Select best candidate
    candidates.sort(key=lambda x: x[0])
    best_predict = candidates[0][1]
    
    def final_predict(X):
        try:
            y = best_predict(X)
            y = np.where(np.isfinite(y), y, 0.0)
            return y
        except Exception:
            return np.zeros(X.shape[0])
            
    return final_predict
# EVOLVE-BLOCK-END
