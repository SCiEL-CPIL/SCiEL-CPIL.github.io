# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
from scipy.interpolate import make_lsq_spline
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    d = X.shape[1]
    n = X.shape[0]
    
    # For 1D problems, we can use 1D-specific methods
    if d == 1:
        x = X[:, 0]
    else:
        x = X  # General case
    
    # Strategy: Try multiple basis families and select the best via CV
    # Basis families: Chebyshev, Fourier, RBF, Spline, Polynomial (high degree)
    
    candidates = []
    
    # 1. Chebyshev polynomial basis
    try:
        if d == 1:
            for deg in [5, 8, 12, 16, 20, 25, 30]:
                if time.time() > deadline:
                    break
                # Scale x to [-1, 1] for Chebyshev
                x_min, x_max = x.min(), x.max()
                x_scaled = 2 * (x - x_min) / (x_max - x_min + 1e-12) - 1
                
                def cheb_features(X_test, deg=deg, x_min=x_min, x_max=x_max):
                    x_t = np.asarray(X_test, dtype=np.float64)
                    if x_t.ndim == 1:
                        x_t = x_t.reshape(-1, 1)
                    x_s = 2 * (x_t[:, 0] - x_min) / (x_max - x_min + 1e-12) - 1
                    # Compute Chebyshev polynomials
                    T = np.zeros((len(x_s), deg + 1))
                    T[:, 0] = 1
                    T[:, 1] = x_s
                    for k in range(2, deg + 1):
                        T[:, k] = 2 * x_s * T[:, k-1] - T[:, k-2]
                    return T
                
                feat_train = cheb_features(x)
                w, *_ = np.linalg.lstsq(feat_train, y, rcond=None)
                pred_train = feat_train @ w
                train_mse = np.mean((pred_train - y) ** 2)
                
                # Quick CV check
                cv_scores = []
                n_folds = 5
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                fold_size = n // n_folds
                for i in range(n_folds):
                    val_idx = indices[i*fold_size:(i+1)*fold_size]
                    tr_idx = np.setdiff1d(indices, val_idx)
                    feat_tr = cheb_features(x[tr_idx])
                    feat_val = cheb_features(x[val_idx])
                    w_cv, *_ = np.linalg.lstsq(feat_tr, y[tr_idx], rcond=None)
                    pred_val = feat_val @ w_cv
                    cv_scores.append(np.mean((pred_val - y[val_idx]) ** 2))
                cv_mse = np.mean(cv_scores)
                
                def predict_cheb(X_test, w=w, deg=deg, x_min=x_min, x_max=x_max):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(-1, 1)
                    return cheb_features(X_t, deg, x_min, x_max) @ w
                
                candidates.append((cv_mse, predict_cheb, f"Chebyshev deg={deg}"))
        else:
            for deg in [3, 5, 7]:
                if time.time() > deadline:
                    break
                # Multivariate Chebyshev - use tensor product
                def multivariate_cheb_features(X_test, deg=deg):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(1, -1)
                    n_pts = X_t.shape[0]
                    d_dim = X_t.shape[1]
                    # Scale each dimension to [-1, 1]
                    mins = X_train.min(axis=0)
                    maxs = X_train.max(axis=0)
                    X_scaled = np.zeros_like(X_t)
                    for j in range(d_dim):
                        X_scaled[:, j] = 2 * (X_t[:, j] - mins[j]) / (maxs[j] - mins[j] + 1e-12) - 1
                    
                    # Generate all monomials up to total degree deg
                    features = [np.ones(n_pts)]
                    for total_deg in range(1, deg + 1):
                        # Generate all compositions of total_deg into d_dim parts
                        def gen_compositions(total, parts):
                            if parts == 1:
                                yield [total]
                            else:
                                for i in range(total + 1):
                                    for rest in gen_compositions(total - i, parts - 1):
                                        yield [i] + rest
                        
                        for comp in gen_compositions(total_deg, d_dim):
                            feat = np.ones(n_pts)
                            for j in range(d_dim):
                                if comp[j] > 0:
                                    # Compute Chebyshev polynomial of order comp[j]
                                    c = comp[j]
                                    T = np.zeros(n_pts)
                                    if c == 0:
                                        T = np.ones(n_pts)
                                    elif c == 1:
                                        T = X_scaled[:, j]
                                    else:
                                        T0 = np.ones(n_pts)
                                        T1 = X_scaled[:, j]
                                        for k in range(2, c + 1):
                                            T2 = 2 * X_scaled[:, j] * T1 - T0
                                            T0, T1 = T1, T2
                                        T = T1
                                    feat = feat * T
                            features.append(feat)
                    return np.column_stack(features)
                
                feat_train = multivariate_cheb_features(X)
                w, *_ = np.linalg.lstsq(feat_train, y, rcond=None)
                
                cv_scores = []
                n_folds = 5
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                fold_size = n // n_folds
                for i in range(n_folds):
                    val_idx = indices[i*fold_size:(i+1)*fold_size]
                    tr_idx = np.setdiff1d(indices, val_idx)
                    feat_tr = multivariate_cheb_features(X[tr_idx])
                    feat_val = multivariate_cheb_features(X[val_idx])
                    w_cv, *_ = np.linalg.lstsq(feat_tr, y[tr_idx], rcond=None)
                    pred_val = feat_val @ w_cv
                    cv_scores.append(np.mean((pred_val - y[val_idx]) ** 2))
                cv_mse = np.mean(cv_scores)
                
                def predict_mv_cheb(X_test, w=w, deg=deg):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(1, -1)
                    return multivariate_cheb_features(X_t, deg) @ w
                
                candidates.append((cv_mse, predict_mv_cheb, f"MV Chebyshev deg={deg}"))
    except Exception:
        pass
    
    # 2. Fourier basis (for 1D)
    try:
        if d == 1:
            for n_harm in [5, 10, 15, 20, 30]:
                if time.time() > deadline:
                    break
                x_min, x_max = x.min(), x.max()
                period = x_max - x_min + 1e-12
                
                def fourier_features(X_test, n_harm=n_harm, x_min=x_min, x_max=x_max):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(-1, 1)
                    x_t = X_t[:, 0]
                    n_pts = len(x_t)
                    cols = [np.ones(n_pts)]
                    for k in range(1, n_harm + 1):
                        freq = 2 * np.pi * k / period
                        cols.append(np.sin(freq * (x_t - x_min)))
                        cols.append(np.cos(freq * (x_t - x_min)))
                    return np.column_stack(cols)
                
                feat_train = fourier_features(x)
                w, *_ = np.linalg.lstsq(feat_train, y, rcond=None)
                
                cv_scores = []
                n_folds = 5
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                fold_size = n // n_folds
                for i in range(n_folds):
                    val_idx = indices[i*fold_size:(i+1)*fold_size]
                    tr_idx = np.setdiff1d(indices, val_idx)
                    feat_tr = fourier_features(x[tr_idx])
                    feat_val = fourier_features(x[val_idx])
                    w_cv, *_ = np.linalg.lstsq(feat_tr, y[tr_idx], rcond=None)
                    pred_val = feat_val @ w_cv
                    cv_scores.append(np.mean((pred_val - y[val_idx]) ** 2))
                cv_mse = np.mean(cv_scores)
                
                def predict_fourier(X_test, w=w, n_harm=n_harm, x_min=x_min, x_max=x_max):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(-1, 1)
                    return fourier_features(X_t, n_harm, x_min, x_max) @ w
                
                candidates.append((cv_mse, predict_fourier, f"Fourier n_harm={n_harm}"))
    except Exception:
        pass
    
    # 3. RBF kernel regression
    try:
        if d == 1:
            x_sorted = np.sort(x)
            for n_rbf in [10, 20, 40, 60]:
                if time.time() > deadline:
                    break
                # Use evenly spaced centers
                centers = np.linspace(x.min(), x.max(), n_rbf)
                
                def rbf_features(X_test, centers=centers, n_rbf=n_rbf):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(-1, 1)
                    x_t = X_t[:, 0]
                    # Gaussian RBF
                    diff = x_t[:, np.newaxis] - centers[np.newaxis, :]
                    # Adaptive bandwidth
                    bandwidth = (x.max() - x.min()) / (n_rbf + 1)
                    K = np.exp(-0.5 * (diff / bandwidth) ** 2)
                    return K
                
                feat_train = rbf_features(x)
                w, *_ = np.linalg.lstsq(feat_train, y, rcond=None)
                
                cv_scores = []
                n_folds = 5
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                fold_size = n // n_folds
                for i in range(n_folds):
                    val_idx = indices[i*fold_size:(i+1)*fold_size]
                    tr_idx = np.setdiff1d(indices, val_idx)
                    feat_tr = rbf_features(x[tr_idx])
                    feat_val = rbf_features(x[val_idx])
                    w_cv, *_ = np.linalg.lstsq(feat_tr, y[tr_idx], rcond=None)
                    pred_val = feat_val @ w_cv
                    cv_scores.append(np.mean((pred_val - y[val_idx]) ** 2))
                cv_mse = np.mean(cv_scores)
                
                def predict_rbf(X_test, w=w, centers=centers, n_rbf=n_rbf):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(-1, 1)
                    return rbf_features(X_t, centers, n_rbf) @ w
                
                candidates.append((cv_mse, predict_rbf, f"RBF n={n_rbf}"))
    except Exception:
        pass
    
    # 4. Smoothing spline
    try:
        if d == 1:
            for s_param in [0.01, 0.1, 1.0, 10.0, 100.0]:
                if time.time() > deadline:
                    break
                # Sort by x for spline
                sort_idx = np.argsort(x)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]
                
                def predict_spline(X_test, x_sorted=x_sorted, y_sorted=y_sorted, s_param=s_param):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(-1, 1)
                    x_t = X_t[:, 0]
                    # Use scipy's make_lsq_spline or UnivariateSpline
                    from scipy.interpolate import UnivariateSpline
                    spl = UnivariateSpline(x_sorted, y_sorted, s=s_param)
                    return spl(x_t)
                
                # Evaluate CV
                cv_scores = []
                n_folds = 5
                indices = np.arange(n)
                np.random.seed(42)
                np.random.shuffle(indices)
                fold_size = n // n_folds
                for i in range(n_folds):
                    val_idx = indices[i*fold_size:(i+1)*fold_size]
                    tr_idx = np.setdiff1d(indices, val_idx)
                    spl = UnivariateSpline(x[tr_idx], y[tr_idx], s=s_param)
                    pred_val = spl(x[val_idx])
                    cv_scores.append(np.mean((pred_val - y[val_idx]) ** 2))
                cv_mse = np.mean(cv_scores)
                
                candidates.append((cv_mse, predict_spline, f"Spline s={s_param}"))
    except Exception:
        pass
    
    # 5. High-degree polynomial with regularization
    try:
        for deg in [8, 12, 16, 20]:
            if time.time() > deadline:
                break
            if d == 1:
                x_min, x_max = x.min(), x.max()
                def poly_features(X_test, deg=deg, x_min=x_min, x_max=x_max):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(-1, 1)
                    x_t = X_t[:, 0]
                    x_s = 2 * (x_t - x_min) / (x_max - x_min + 1e-12) - 1
                    cols = [x_s ** k for k in range(deg + 1)]
                    return np.column_stack(cols)
            else:
                def poly_features(X_test, deg=deg):
                    X_t = np.asarray(X_test, dtype=np.float64)
                    if X_t.ndim == 1:
                        X_t = X_t.reshape(1, -1)
                    n_pts = X_t.shape[0]
                    d_dim = X_t.shape[1]
                    cols = [np.ones(n_pts)]
                    for total_deg in range(1, deg + 1):
                        def gen_compositions(total, parts):
                            if parts == 1:
                                yield [total]
                            else:
                                for i in range(total + 1):
                                    for rest in gen_compositions(total - i, parts - 1):
                                        yield [i] + rest
                        for comp in gen_compositions(total_deg, d_dim):
                            feat = np.ones(n_pts)
                            for j in range(d_dim):
                                if comp[j] > 0:
                                    feat = feat * X_t[:, j] ** comp[j]
                            cols.append(feat)
                    return np.column_stack(cols)
            
            feat_train = poly_features(X)
            # Regularized least squares
            n_feat = feat_train.shape[1]
            lam = 1e-4
            w = np.linalg.solve(feat_train.T @ feat_train + lam * np.eye(n_feat), feat_train.T @ y)
            
            cv_scores = []
            n_folds = 5
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            fold_size = n // n_folds
            for i in range(n_folds):
                val_idx = indices[i*fold_size:(i+1)*fold_size]
                tr_idx = np.setdiff1d(indices, val_idx)
                feat_tr = poly_features(X[tr_idx])
                feat_val = poly_features(X[val_idx])
                n_feat_tr = feat_tr.shape[1]
                w_cv = np.linalg.solve(feat_tr.T @ feat_tr + lam * np.eye(n_feat_tr), feat_tr.T @ y[tr_idx])
                pred_val = feat_val @ w_cv
                cv_scores.append(np.mean((pred_val - y[val_idx]) ** 2))
            cv_mse = np.mean(cv_scores)
            
            def predict_poly(X_test, w=w, deg=deg):
                X_t = np.asarray(X_test, dtype=np.float64)
                if X_t.ndim == 1:
                    X_t = X_t.reshape(1, -1)
                return poly_features(X_t, deg) @ w
            
            candidates.append((cv_mse, predict_poly, f"Poly deg={deg}"))
    except Exception:
        pass
    
    # Select best candidate
    if not candidates:
        # Fallback: simple linear regression
        def fallback_predict(X_test):
            X_t = np.asarray(X_test, dtype=np.float64)
            if X_t.ndim == 1:
                X_t = X_t.reshape(1, -1)
            A = np.column_stack([np.ones(X_t.shape[0]), X_t])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            return A @ w
        return fallback_predict
    
    candidates.sort(key=lambda x: x[0])
    best_cv_mse, best_predict, best_name = candidates[0]
    
    # Final sanity check: ensure predictions are finite
    test_X = X[:10]
    test_pred = best_predict(test_X)
    if not np.all(np.isfinite(test_pred)):
        # Try next candidate
        for cv_mse, predict_fn, name in candidates[1:]:
            try:
                test_pred = predict_fn(test_X)
                if np.all(np.isfinite(test_pred)):
                    return predict_fn
            except Exception:
                continue
        # Fallback
        def fallback_predict(X_test):
            X_t = np.asarray(X_test, dtype=np.float64)
            if X_t.ndim == 1:
                X_t = X_t.reshape(1, -1)
            A = np.column_stack([np.ones(X_t.shape[0]), X_t])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            return A @ w
        return fallback_predict
    
    return best_predict
# EVOLVE-BLOCK-END
