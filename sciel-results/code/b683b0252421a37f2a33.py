import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 275  # Safety margin before 300s limit
    
    X = X_train
    y = y_train
    n, d = X.shape
    x = X[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    var_y = np.var(y)
    if var_y < 1e-15:
        return lambda X: np.zeros(len(X))
    
    # Strategy 1: Cubic spline interpolation (natural boundary)
    cs = None
    try:
        cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')
        spline_pred = cs(x_sorted)
        if not np.all(np.isfinite(spline_pred)):
            cs = None
    except Exception:
        cs = None
    
    # Strategy 2: PCHIP
    pchip = None
    try:
        pchip = PchipInterpolator(x_sorted, y_sorted)
        pchip_pred = pchip(x_sorted)
        if not np.all(np.isfinite(pchip_pred)):
            pchip = None
    except Exception:
        pchip = None
    
    # Strategy 3: RBF with Gaussian kernel
    rbf_w = None
    rbf_centers = None
    rbf_width = None
    try:
        # Use a subset of centers to keep matrix size manageable
        n_centers = min(n, 50)
        if n_centers >= 2:
            center_idx = np.linspace(0, n - 1, n_centers, dtype=int)
            centers = x_sorted[center_idx]
            
            # Estimate width from data spread
            x_range = x_sorted[-1] - x_sorted[0]
            if x_range < 1e-10:
                x_range = 1.0
            # Try multiple widths
            widths = [x_range * w for w in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]]
            
            best_rbf_err = np.inf
            best_rbf_w = None
            best_rbf_width = None
            
            for width in widths:
                # Build kernel matrix
                diff = x_sorted[:, None] - centers[None, :]
                K = np.exp(-0.5 * (diff / width) ** 2)
                
                # Add small regularization
                reg = 1e-8
                A = K.T @ K + reg * np.eye(K.shape[1])
                b = K.T @ y_sorted
                
                try:
                    w = np.linalg.solve(A, b)
                    pred = K @ w
                    if np.all(np.isfinite(pred)):
                        err = np.mean((pred - y_sorted) ** 2)
                        if err < best_rbf_err:
                            best_rbf_err = err
                            best_rbf_w = w
                            best_rbf_width = width
                except Exception:
                    continue
            
            if best_rbf_w is not None:
                rbf_w = best_rbf_w
                rbf_centers = centers
                rbf_width = best_rbf_width
    except Exception:
        rbf_w = None
        
    # Strategy 4: Dense RBF (using all points as centers) with ridge regression
    # This acts as a dense non-parametric interpolant, avoiding oscillation of splines
    dense_rbf_w = None
    dense_rbf_centers = None
    dense_rbf_width = None
    try:
        if n >= 2:
            centers = x_sorted
            x_range = x_sorted[-1] - x_sorted[0]
            if x_range < 1e-10:
                x_range = 1.0
            
            # Try a few widths for the dense RBF
            # Since we have n centers, we need to solve n x n system. n=200 is fine.
            widths_dense = [x_range * w for w in [0.05, 0.1, 0.2, 0.5]]
            
            best_dense_err = np.inf
            best_dense_w = None
            best_dense_width = None
            
            for width in widths_dense:
                diff = x_sorted[:, None] - centers[None, :]
                K = np.exp(-0.5 * (diff / width) ** 2)
                
                # Use ridge regression for stability
                # alpha controls the trade-off between fitting the data and minimizing weights
                # Since data is noiseless, we want low alpha (interpolation), but some regularization helps
                # Try a range of alphas
                alphas = [1e-10, 1e-8, 1e-6, 1e-4]
                
                for alpha in alphas:
                    A = K.T @ K + alpha * np.eye(K.shape[1])
                    b = K.T @ y_sorted
                    
                    try:
                        w = np.linalg.solve(A, b)
                        pred = K @ w
                        if np.all(np.isfinite(pred)):
                            err = np.mean((pred - y_sorted) ** 2)
                            # Penalize large weights slightly to prefer smoother solutions if error is similar
                            # But primarily minimize error
                            if err < best_dense_err:
                                best_dense_err = err
                                best_dense_w = w
                                best_dense_width = width
                    except Exception:
                        continue
            
            if best_dense_w is not None:
                dense_rbf_w = best_dense_w
                dense_rbf_centers = centers
                dense_rbf_width = best_dense_width
    except Exception:
        dense_rbf_w = None

    # Strategy 5: Rich basis + ridge regression
    def make_features(x):
        p1 = np.ones_like(x)
        p2 = x
        p3 = x**2
        p4 = x**3
        p5 = x**4
        p6 = x**5
        p7 = x**6
        t1 = np.sin(x)
        t2 = np.cos(x)
        t3 = np.sin(2*x)
        t4 = np.cos(2*x)
        t5 = np.sin(3*x)
        t6 = np.cos(3*x)
        t7 = np.sin(4*x)
        t8 = np.cos(4*x)
        t9 = np.sin(5*x)
        t10 = np.cos(5*x)
        t11 = np.sin(6*x)
        t12 = np.cos(6*x)
        e1 = np.exp(x)
        e2 = np.exp(-x)
        e3 = np.exp(2*x)
        e4 = np.exp(-2*x)
        r1 = np.sqrt(np.abs(x) + 1e-10)
        r2 = np.log(np.abs(x) + 1e-10)
        r3 = 1.0 / (x + 1e-10)
        r4 = 1.0 / (x**2 + 1e-10)
        return np.column_stack([p1, p2, p3, p4, p5, p6, p7,
                                  t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12,
                                  e1, e2, e3, e4,
                                  r1, r2, r3, r4])
    
    ridge_w = None
    try:
        F_train = make_features(x)
        alpha = 1e-6
        A = F_train.T @ F_train + alpha * np.eye(F_train.shape[1])
        b = F_train.T @ y
        ridge_w = np.linalg.solve(A, b)
        ridge_pred = F_train @ ridge_w
        if not np.all(np.isfinite(ridge_pred)):
            ridge_w = None
    except Exception:
        ridge_w = None
    
    # Cross-validation to pick the best strategy
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(20, n // 5)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    
    x_val = x[val_idx]
    y_val = y[val_idx]
    x_tr = x[tr_idx]
    y_tr = y[tr_idx]
    
    # Evaluate spline
    spline_cv_err = np.inf
    if cs is not None:
        try:
            cs_tr = CubicSpline(x_tr, y_tr, bc_type='natural')
            pred = cs_tr(x_val)
            if np.all(np.isfinite(pred)):
                spline_cv_err = np.mean((pred - y_val) ** 2)
        except Exception:
            pass
    
    # Evaluate pchip
    pchip_cv_err = np.inf
    if pchip is not None:
        try:
            pchip_tr = PchipInterpolator(x_tr, y_tr)
            pred = pchip_tr(x_val)
            if np.all(np.isfinite(pred)):
                pchip_cv_err = np.mean((pred - y_val) ** 2)
        except Exception:
            pass
    
    # Evaluate RBF (subset)
    rbf_cv_err = np.inf
    if rbf_w is not None:
        try:
            # Rebuild RBF on training subset
            n_centers_tr = min(len(x_tr), 50)
            if n_centers_tr >= 2:
                center_idx_tr = np.linspace(0, len(x_tr) - 1, n_centers_tr, dtype=int)
                x_tr_sorted = np.sort(x_tr)
                y_tr_sorted = y_tr[np.argsort(x_tr)]
                centers_tr = x_tr_sorted[center_idx_tr]
                
                x_range_tr = x_tr_sorted[-1] - x_tr_sorted[0]
                if x_range_tr < 1e-10:
                    x_range_tr = 1.0
                widths_tr = [x_range_tr * w for w in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]]
                
                best_rbf_tr_err = np.inf
                best_rbf_tr_w = None
                best_rbf_tr_width = None
                
                for width in widths_tr:
                    diff_tr = x_tr_sorted[:, None] - centers_tr[None, :]
                    K_tr = np.exp(-0.5 * (diff_tr / width) ** 2)
                    reg = 1e-8
                    A_tr = K_tr.T @ K_tr + reg * np.eye(K_tr.shape[1])
                    b_tr = K_tr.T @ y_tr_sorted
                    try:
                        w_tr = np.linalg.solve(A_tr, b_tr)
                        pred_tr = K_tr @ w_tr
                        if np.all(np.isfinite(pred_tr)):
                            err_tr = np.mean((pred_tr - y_tr_sorted) ** 2)
                            if err_tr < best_rbf_tr_err:
                                best_rbf_tr_err = err_tr
                                best_rbf_tr_w = w_tr
                                best_rbf_tr_width = width
                    except Exception:
                        continue
                
                if best_rbf_tr_w is not None:
                    diff_val = x_val[:, None] - centers_tr[None, :]
                    K_val = np.exp(-0.5 * (diff_val / best_rbf_tr_width) ** 2)
                    pred = K_val @ best_rbf_tr_w
                    if np.all(np.isfinite(pred)):
                        rbf_cv_err = np.mean((pred - y_val) ** 2)
        except Exception:
            pass
            
    # Evaluate Dense RBF
    dense_rbf_cv_err = np.inf
    if dense_rbf_w is not None:
        try:
            x_tr_sorted = np.sort(x_tr)
            y_tr_sorted = y_tr[np.argsort(x_tr)]
            centers_tr = x_tr_sorted
            x_range_tr = x_tr_sorted[-1] - x_tr_sorted[0]
            if x_range_tr < 1e-10:
                x_range_tr = 1.0
            
            widths_dense_tr = [x_range_tr * w for w in [0.05, 0.1, 0.2, 0.5]]
            best_dense_tr_err = np.inf
            best_dense_tr_w = None
            best_dense_tr_width = None
            
            for width in widths_dense_tr:
                diff_tr = x_tr_sorted[:, None] - centers_tr[None, :]
                K_tr = np.exp(-0.5 * (diff_tr / width) ** 2)
                alphas = [1e-10, 1e-8, 1e-6, 1e-4]
                for alpha in alphas:
                    A_tr = K_tr.T @ K_tr + alpha * np.eye(K_tr.shape[1])
                    b_tr = K_tr.T @ y_tr_sorted
                    try:
                        w_tr = np.linalg.solve(A_tr, b_tr)
                        pred_tr = K_tr @ w_tr
                        if np.all(np.isfinite(pred_tr)):
                            err_tr = np.mean((pred_tr - y_tr_sorted) ** 2)
                            if err_tr < best_dense_tr_err:
                                best_dense_tr_err = err_tr
                                best_dense_tr_w = w_tr
                                best_dense_tr_width = width
                    except Exception:
                        continue
            
            if best_dense_tr_w is not None:
                diff_val = x_val[:, None] - centers_tr[None, :]
                K_val = np.exp(-0.5 * (diff_val / best_dense_tr_width) ** 2)
                pred = K_val @ best_dense_tr_w
                if np.all(np.isfinite(pred)):
                    dense_rbf_cv_err = np.mean((pred - y_val) ** 2)
        except Exception:
            pass

    # Evaluate ridge
    ridge_cv_err = np.inf
    if ridge_w is not None:
        try:
            F_tr = make_features(x_tr)
            alpha = 1e-6
            A = F_tr.T @ F_tr + alpha * np.eye(F_tr.shape[1])
            b = F_tr.T @ y_tr
            w_tr = np.linalg.solve(A, b)
            F_val = make_features(x_val)
            pred = F_val @ w_tr
            if np.all(np.isfinite(pred)):
                ridge_cv_err = np.mean((pred - y_val) ** 2)
        except Exception:
            pass
    
    # Pick best strategy
    strategies = []
    if cs is not None:
        strategies.append(('spline', spline_cv_err, cs))
    if pchip is not None:
        strategies.append(('pchip', pchip_cv_err, pchip))
    if rbf_w is not None:
        strategies.append(('rbf', rbf_cv_err, (rbf_centers, rbf_width, rbf_w)))
    if dense_rbf_w is not None:
        strategies.append(('dense_rbf', dense_rbf_cv_err, (dense_rbf_centers, dense_rbf_width, dense_rbf_w)))
    if ridge_w is not None:
        strategies.append(('ridge', ridge_cv_err, ridge_w))
    
    if not strategies:
        mean_y = np.mean(y)
        return lambda X: np.full(len(X), mean_y)
    
    strategies.sort(key=lambda s: s[1])
    best_name, best_err, best_obj = strategies[0]
    
    if best_name == 'spline':
        def predict(X):
            xq = X[:, 0]
            return best_obj(xq)
        return predict
    elif best_name == 'pchip':
        def predict(X):
            xq = X[:, 0]
            return best_obj(xq)
        return predict
    elif best_name == 'rbf':
        centers, width, w = best_obj
        def predict(X):
            xq = X[:, 0]
            diff = xq[:, None] - centers[None, :]
            K = np.exp(-0.5 * (diff / width) ** 2)
            return K @ w
        return predict
    elif best_name == 'dense_rbf':
        centers, width, w = best_obj
        def predict(X):
            xq = X[:, 0]
            diff = xq[:, None] - centers[None, :]
            K = np.exp(-0.5 * (diff / width) ** 2)
            return K @ w
        return predict
    else:  # ridge
        def predict(X):
            xq = X[:, 0]
            F = make_features(xq)
            return F @ best_obj
        return predict
