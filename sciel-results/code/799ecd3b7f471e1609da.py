# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import make_interp_spline, CubicSpline, PchipInterpolator

_TOTAL_DEADLINE = time.time() + 318.0


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(make_fn):
    """Wrap a prediction function to always return finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict


def _poly_features(X, degree):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
    if d >= 2:
        for i in range(d):
            for j in range(i, d):
                for p in range(1, degree + 1):
                    for q in range(1, degree + 1 - p + 1):
                        if p + q <= degree:
                            cols.append((X[:, i] ** p) * (X[:, j] ** q))
    return np.column_stack(cols)


def _trig_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        for k in range(1, 8):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _exp_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        safe_x = np.where(np.abs(x) < 1e-9, 1e-9 * np.sign(x + 1e-15), x)
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(1.0 / (x + 1e-9))
        cols.append(x * np.exp(-x))
        cols.append(x * np.exp(x))
        cols.append(np.exp(-0.5 * x * x))
    return np.column_stack(cols)


def _mixed_features(X):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for j in range(d):
        x = X[:, j]
        cols.append(x)
        cols.append(x ** 2)
        cols.append(x ** 3)
        cols.append(x ** 4)
        cols.append(x ** 5)
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.sin(2 * x))
        cols.append(np.cos(2 * x))
        cols.append(np.sin(3 * x))
        cols.append(np.cos(3 * x))
        cols.append(np.tanh(x))
        cols.append(np.exp(-x ** 2))
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(np.exp(-0.5 * x * x))
    return np.column_stack(cols)


def _fit_ols(F, y, ridge_alpha=1e-6):
    try:
        n, p = F.shape
        if ridge_alpha > 0:
            F_reg = F.copy()
            F_reg[:, 0] += ridge_alpha * np.sqrt(n)
            w, *_ = np.linalg.lstsq(F_reg, y, rcond=None)
        else:
            w, *_ = np.linalg.lstsq(F, y, rcond=None)
        if not np.all(np.isfinite(w)):
            return None
        return w
    except Exception:
        return None


def _rmse(F, w, y):
    pred = F @ w
    return np.sqrt(np.mean((pred - y) ** 2))


def _cv_spline_score(X, y, num_knots, k=3, n_folds=5):
    """Cross-validate a spline with given number of knots. Returns CV MSE."""
    try:
        if num_knots < k + 2:
            return np.inf
        x = X[:, 0].copy()
        ys = y.copy()
        n = len(x)
        
        # Sort by x
        order = np.argsort(x)
        xs = x[order]
        ysv = ys[order]
        
        # Check for duplicate x values
        unique_x, unique_indices = np.unique(xs, return_index=True)
        if len(unique_x) < num_knots + k:
            # Too few unique points
            return np.inf
        
        # K-fold CV
        fold_size = n // n_folds
        fold_starts = np.linspace(0, n, n_folds + 1, dtype=int)
        
        cv_errors = []
        for fold in range(n_folds):
            start = fold_starts[fold]
            end = fold_starts[fold + 1]
            test_idx = np.arange(start, end)
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            
            if len(train_idx) < num_knots + k:
                return np.inf
            
            xtr = xs[train_idx]
            ytr = ysv[train_idx]
            xte = xs[test_idx]
            yte = ysv[test_idx]
            
            # Select knot positions from training data
            if num_knots <= len(xtr):
                knot_idx = np.linspace(0, len(xtr) - 1, num_knots, dtype=int)
                knot_x = xtr[knot_idx]
                knot_y = ytr[knot_idx]
                
                # Ensure unique knot positions
                knot_x_unique, knot_unique_idx = np.unique(knot_x, return_index=True)
                knot_y_unique = knot_y[knot_unique_idx]
                
                if len(knot_x_unique) < k + 2:
                    return np.inf
                
                try:
                    s = make_interp_spline(knot_x_unique, knot_y_unique, k=k)
                    pred_te = s(xte)
                    mse = np.mean((pred_te - yte) ** 2)
                    if not np.isfinite(mse):
                        return np.inf
                    cv_errors.append(mse)
                except Exception:
                    return np.inf
        
        if not cv_errors:
            return np.inf
        return np.mean(cv_errors)
    except Exception:
        return np.inf


def _fit_spline_cv(X, y, min_knots=5, max_knots=40, k=3):
    """Fit a cubic spline with CV-selected knot count."""
    try:
        x = X[:, 0].copy()
        ys = y.copy()
        n = len(x)
        
        order = np.argsort(x)
        xs = x[order]
        ysv = ys[order]
        
        # Check for duplicates
        unique_x = np.unique(xs)
        if len(unique_x) < min_knots + k:
            return None
        
        best_knots = None
        best_cv = np.inf
        
        # Try various knot counts
        knot_candidates = list(range(min_knots, min(max_knots + 1, n - k)))
        # Add some strategic values
        for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            nk = int(frac * n)
            if min_knots <= nk <= max_knots and nk not in knot_candidates:
                knot_candidates.append(nk)
        knot_candidates = sorted(set(knot_candidates))
        
        for nk in knot_candidates:
            if _remaining() < 5.0:
                break
            cv = _cv_spline_score(X, y, nk, k=k)
            if cv < best_cv:
                best_cv = cv
                best_knots = nk
        
        if best_knots is None or best_cv >= np.inf:
            return None
        
        # Fit final spline on all data with best knot count
        knot_idx = np.linspace(0, n - 1, best_knots, dtype=int)
        knot_x = xs[knot_idx]
        knot_y = ysv[knot_idx]
        
        knot_x_unique, knot_unique_idx = np.unique(knot_x, return_index=True)
        knot_y_unique = knot_y[knot_unique_idx]
        
        if len(knot_x_unique) < k + 2:
            return None
        
        s = make_interp_spline(knot_x_unique, knot_y_unique, k=k)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = s(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_pchip(X, y):
    """Fit a PCHIP (shape-preserving) interpolant."""
    try:
        x = X[:, 0].copy()
        ys = y.copy()
        n = len(x)
        
        order = np.argsort(x)
        xs = x[order]
        ysv = ys[order]
        
        # Handle duplicates by averaging
        unique_x, inverse = np.unique(xs, return_inverse=True)
        if len(unique_x) < 4:
            return None
        
        # Average y values for duplicate x
        y_unique = np.zeros(len(unique_x))
        counts = np.zeros(len(unique_x))
        np.add.at(y_unique, inverse, ysv)
        np.add.at(counts, inverse, 1)
        y_unique /= counts
        
        if len(unique_x) < 4:
            return None
        
        pchip = PchipInterpolator(unique_x, y_unique)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = pchip(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None


def _fit_gaussian_process(X, y, length_scales=None, noise=1e-6):
    """Simple Gaussian Process regression with multiple length scales."""
    try:
        n, d = X.shape
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        
        if length_scales is None:
            length_scales = np.ones(d)
        
        def kernel(X1, X2):
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2) / (length_scales[np.newaxis, np.newaxis, :]**2)
            return np.exp(-0.5 * sq_dist)
        
        K = kernel(Xn, Xn) + noise * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except:
            K += 1e-4 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except:
                return None
        
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            Ks = kernel(Xpn, Xn)
            preds = Ks @ alpha
            return preds.flatten()
        return predict
    except Exception:
        return None


def _fit_rbf(X, y, num_centers=25, num_widths=5):
    """Fit RBF network with learned centers and widths."""
    try:
        n, d = X.shape
        if n < num_centers + 4:
            num_centers = max(5, n - 4)
        
        np.random.seed(123)
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        centers = x_min + np.random.rand(num_centers, d) * (x_max - x_min)
        widths = np.random.uniform(0.1, 1.0, num_widths)
        
        def objective(params):
            if _remaining() < 2.0:
                return 1e12
            c = params[:num_centers * d].reshape(num_centers, d)
            w = params[num_centers * d:]
            F = _gaussian_rbf_features(X, c, w)
            try:
                ww, *_ = np.linalg.lstsq(F, y, rcond=None)
                pred = F @ ww
                return np.mean((pred - y)**2)
            except:
                return 1e12
        
        bounds = [(-1, 1)] * (num_centers * d) + [(0.01, 2.0)] * num_widths
        result = differential_evolution(objective, bounds, maxiter=50, popsize=10, tol=1e-4, seed=42)
        
        if result.fun < 1e10:
            best_params = result.x
            c = best_params[:num_centers * d].reshape(num_centers, d)
            w = best_params[num_centers * d:]
            F = _gaussian_rbf_features(X, c, w)
            ww, *_ = np.linalg.lstsq(F, y, rcond=None)
            
            def predict(Xp):
                Xp = np.asarray(Xp, dtype=np.float64)
                if Xp.ndim == 1:
                    Xp = Xp.reshape(1, -1)
                Fp = _gaussian_rbf_features(Xp, c, w)
                return Fp @ ww
            return _safe_predict(predict)
    except Exception:
        return None


def _gaussian_rbf_features(X, centers, widths):
    d = X.shape[1]
    cols = [np.ones(X.shape[0])]
    for c in centers:
        for w in widths:
            dist = np.sqrt(np.sum((X - c) ** 2, axis=1))
            cols.append(np.exp(-0.5 * (dist / w) ** 2))
    return np.column_stack(cols)


def _fit_mlp(X, y, hidden_sizes=(64, 64), lr=0.01, epochs=500, seed=42):
    """Small MLP via Adam optimizer."""
    try:
        d = X.shape[1]
        n = X.shape[0]
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        Xn = (X - mu) / sigma
        y_mu = y.mean()
        y_sigma = y.std() + 1e-8
        yn = (y - y_mu) / y_sigma

        np.random.seed(seed)
        layers = [d] + list(hidden_sizes) + [1]
        W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i]) for i in range(len(layers)-1)]
        b = [np.zeros(layers[i+1]) for i in range(len(layers)-1)]
        
        mW = [np.zeros_like(w) for w in W]
        vW = [np.zeros_like(w) for w in W]
        mb = [np.zeros_like(bb) for bb in b]
        vb = [np.zeros_like(bb) for bb in b]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def forward(Xb):
            a = Xb
            acts = [a]
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                if i < len(W) - 1:
                    a = np.tanh(z)
                else:
                    a = z
                acts.append(a)
            return acts

        for epoch in range(epochs):
            if _remaining() < 5.0:
                break
            t += 1
            acts = forward(Xn)
            pred = acts[-1]
            dz = (pred - yn) / n
            
            grads_W = [None] * len(W)
            grads_b = [None] * len(b)
            for i in range(len(W) - 1, -1, -1):
                if i == len(W) - 1:
                    dz_i = dz
                else:
                    dz_i = dz @ W[i].T * (1 - acts[i+1]**2)
                grads_W[i] = acts[i].T @ dz_i
                grads_b[i] = dz_i.sum(axis=0)
            
            for i in range(len(W)):
                mW[i] = beta1 * mW[i] + (1 - beta1) * grads_W[i]
                vW[i] = beta2 * vW[i] + (1 - beta2) * grads_W[i]**2
                mW_hat = mW[i] / (1 - beta1**t)
                vW_hat = vW[i] / (1 - beta2**t)
                W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                
                mb[i] = beta1 * mb[i] + (1 - beta1) * grads_b[i]
                vb[i] = beta2 * vb[i] + (1 - beta2) * grads_b[i]**2
                mb_hat = mb[i] / (1 - beta1**t)
                vb_hat = vb[i] / (1 - beta2**t)
                b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            Xpn = (Xp - mu) / sigma
            a = Xpn
            for i in range(len(W)):
                z = a @ W[i] + b[i]
                a = np.tanh(z) if i < len(W) - 1 else z
            return a.flatten() * y_sigma + y_mu
        return predict
    except Exception:
        return None


def _try_fit_family(feat_fn, X, y, ridge_alpha=1e-6):
    try:
        F = feat_fn(X)
        if not np.all(np.isfinite(F)):
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        w = _fit_ols(F, y, ridge_alpha)
        if w is None:
            return None, np.inf
        rmse = _rmse(F, w, y)
        predict = _safe_predict(lambda Xp: feat_fn(Xp) @ w)
        return predict, rmse
    except Exception:
        return None, np.inf


def _polish_with_local_opt(X, y, initial_predict, n_iter=50):
    """Polish predictions using local optimization on residuals."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        
        # Get initial predictions
        init_pred = initial_predict(X)
        init_pred_sorted = init_pred[order]
        
        # Compute residuals
        residuals = ys - init_pred_sorted
        
        # Use a smooth basis to fit residuals
        # Try a few RBF centers
        np.random.seed(77)
        n_centers = min(20, len(xs) // 3)
        if n_centers < 3:
            return initial_predict
        
        centers = np.random.choice(xs, size=n_centers, replace=False)
        centers = centers.reshape(-1, 1)
        widths = np.linspace(0.1, 1.0, 3)
        
        def rbf_feat(Xp):
            cols = []
            for c in centers.flatten():
                for w in widths:
                    cols.append(np.exp(-0.5 * ((Xp[:, 0] - c) / w) ** 2))
            return np.column_stack(cols)
        
        F_res = rbf_feat(X)
        w_res, *_ = np.linalg.lstsq(F_res, residuals, rcond=None)
        
        def polished_predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            base = initial_predict(Xp)
            Fp = rbf_feat(Xp)
            res_pred = Fp @ w_res
            return base + res_pred
        
        # Check if polishing improved things
        polished_train = polished_predict(X)
        rmse_polished = np.sqrt(np.mean((polished_train - y) ** 2))
        rmse_init = np.sqrt(np.mean((init_pred - y) ** 2))
        
        if rmse_polished < rmse_init:
            return polished_predict
        return initial_predict
    except Exception:
        return initial_predict


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    best_predict = None
    best_rmse = np.inf

    # Baseline: constant
    const = np.mean(y_train)
    const_rmse = np.sqrt(np.mean((y_train - const) ** 2))
    best_rmse = const_rmse
    best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    # Try polynomial degrees
    for deg in [1, 2, 3, 4, 5, 6, 7, 8]:
        if _remaining() < 10.0:
            break
        predict, rmse = _try_fit_family(lambda Xp, d=deg: _poly_features(Xp, d), X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try trig
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_trig_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try exp/log
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_exp_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try mixed
    if _remaining() > 10.0:
        predict, rmse = _try_fit_family(_mixed_features, X_train, y_train)
        if predict is not None and rmse < best_rmse:
            best_rmse = rmse
            best_predict = predict

    # Try CV spline (key improvement for 1-D noiseless data)
    if _remaining() > 15.0 and d == 1:
        sp_predict = _fit_spline_cv(X_train, y_train)
        if sp_predict is not None:
            pred = sp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = sp_predict

    # Try PCHIP
    if _remaining() > 10.0 and d == 1:
        pchip_predict = _fit_pchip(X_train, y_train)
        if pchip_predict is not None:
            pred = pchip_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = pchip_predict

    # Try Gaussian Process
    if _remaining() > 15.0:
        gp_predict = _fit_gaussian_process(X_train, y_train)
        if gp_predict is not None:
            pred = gp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = gp_predict

    # Try RBF with optimization
    if _remaining() > 30.0:
        rbf_predict = _fit_rbf(X_train, y_train)
        if rbf_predict is not None:
            pred = rbf_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = rbf_predict

    # Try MLP
    if _remaining() > 30.0:
        mlp_predict = _fit_mlp(X_train, y_train)
        if mlp_predict is not None:
            pred = mlp_predict(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = mlp_predict

    # Polish step
    if _remaining() > 10.0 and best_predict is not None:
        polished = _polish_with_local_opt(X_train, y_train, best_predict)
        if polished is not None:
            pred = polished(X_train)
            rmse = np.sqrt(np.mean((pred - y_train) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_predict = polished

    # Final safety check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(Xp.shape[0], const))

    return best_predict
# EVOLVE-BLOCK-END
