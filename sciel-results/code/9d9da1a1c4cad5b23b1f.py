# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings("ignore")

_TOTAL_DEADLINE = time.time() + 320.0  # safety margin under 330s


def _remaining():
    return _TOTAL_DEADLINE - time.time()


def _safe_predict(feat_fn, w):
    """Build a predict callable that always returns finite values."""
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            F = feat_fn(X)
            if not np.all(np.isfinite(F)):
                F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
            y = F @ w
            y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9)
            return y
        except Exception:
            return np.zeros(X.shape[0])
    return predict


def _mse(y_pred, y_true):
    d = y_pred - y_true
    return float(np.mean(d * d))


def _nrmse(y_pred, y_true):
    v = float(np.var(y_true))
    if v < 1e-12:
        v = 1.0
    return _mse(y_pred, y_true) / v


def _fit_linear(F, y):
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return w


def _poly_features(X, max_deg=8):
    n, d = X.shape
    cols = [np.ones(n)]
    for deg in range(1, max_deg + 1):
        for i in range(d):
            cols.append((X[:, i] ** deg))
    return np.column_stack(cols)


def _trig_features(X, max_freq=6):
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        x = X[:, i]
        for f in range(1, max_freq + 1):
            cols.append(np.sin(f * x))
            cols.append(np.cos(f * x))
    return np.column_stack(cols)


def _exp_features(X, scales=(0.5, 1.0, 1.5, 2.0)):
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        x = X[:, i]
        for s in scales:
            cols.append(np.exp(s * x))
            cols.append(np.exp(-s * x))
        cols.append(np.log(np.abs(x) + 1e-8))
        cols.append(1.0 / (np.abs(x) + 1e-3))
    return np.column_stack(cols)


def _mixed_features(X):
    """Combine polynomial, trig, and some non-polynomial terms."""
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        x = X[:, i]
        for deg in range(1, 7):
            cols.append(x ** deg)
        for f in range(1, 5):
            cols.append(np.sin(f * x))
            cols.append(np.cos(f * x))
        cols.append(np.exp(x))
        cols.append(np.exp(-x))
        cols.append(np.log(np.abs(x) + 1e-8))
        cols.append(1.0 / (np.abs(x) + 1e-3))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
    return np.column_stack(cols)


def _spline_fit(X, y, k=5):
    """Piecewise polynomial interpolation via spline on sorted x."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] == 1:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        # remove duplicate x
        keep = np.concatenate(([True], np.diff(xs) > 1e-12))
        xs, ys = xs[keep], ys[keep]
        if len(xs) < k + 1:
            k = max(1, len(xs) - 1)
        try:
            sp = make_interp_spline(xs, ys, k=k)
            def predict(Xq):
                Xq = np.asarray(Xq, dtype=np.float64)
                if Xq.ndim == 1:
                    Xq = Xq.reshape(1, -1)
                return sp(Xq[:, 0])
            return predict
        except Exception:
            return None
    return None


def _fit_nn(X, y, time_budget=10.0):
    """Small neural net with gradient descent via scipy minimize."""
    n, d = X.shape
    hidden = 32
    # Normalize inputs
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    y_mean = y.mean()
    y_std = y.std() + 1e-8

    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    rng = np.random.default_rng(42)
    W1 = rng.normal(0, 1.0 / np.sqrt(d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
    b2 = np.zeros(1)

    def forward(params):
        W1 = params[:d * hidden].reshape(d, hidden)
        b1 = params[d * hidden:d * hidden + hidden]
        W2 = params[d * hidden + hidden:d * hidden + hidden + hidden].reshape(hidden, 1)
        b2 = params[d * hidden + hidden + hidden:]
        h = np.tanh(Xn @ W1 + b1)
        out = h @ W2 + b2
        return out.flatten(), h, W1, b1, W2, b2

    def loss(params):
        out, _, _, _, _, _ = forward(params)
        return float(np.mean((out - yn) ** 2))

    def grad(params):
        eps = 1e-7
        g = np.zeros_like(params)
        base = loss(params)
        for i in range(len(params)):
            params[i] += eps
            lp = loss(params)
            params[i] -= 2 * eps
            lm = loss(params)
            params[i] += eps
            g[i] = (lp - lm) / (2 * eps)
        return g

    params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
    t0 = time.time()
    best_params = params.copy()
    best_loss = loss(params)
    try:
        res = minimize(loss, params, method='L-BFGS-B', jac=grad,
                       options={'maxiter': 200, 'maxfun': 5000, 'maxls': 50})
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_params = res.x
    except Exception:
        pass

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        Xqn = (Xq - x_mean) / x_std
        W1 = best_params[:d * hidden].reshape(d, hidden)
        b1 = best_params[d * hidden:d * hidden + hidden]
        W2 = best_params[d * hidden + hidden:d * hidden + hidden + hidden].reshape(hidden, 1)
        b2 = best_params[d * hidden + hidden + hidden:]
        h = np.tanh(Xqn @ W1 + b1)
        out = (h @ W2 + b2).flatten()
        out = out * y_std + y_mean
        return np.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)
    return predict, best_loss


def _gaussian_process_fit(X, y, time_budget=8.0):
    """Simple GP regression with RBF kernel, optimized length-scale."""
    n, d = X.shape
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    Xn = (X - x_mean) / x_std
    y_mean = y.mean()
    yn = (y - y_mean) / (y.std() + 1e-8)

    def rbf(A, B, ls):
        # squared distance
        sq = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T
        sq = np.maximum(sq, 0)
        return np.exp(-sq / (2 * ls ** 2))

    def log_marginal(params):
        ls, sigma = params
        K = rbf(Xn, Xn, ls) + sigma ** 2 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            lml = -0.5 * yn @ alpha - np.sum(np.log(np.diag(L)))
            return -float(lml)
        except Exception:
            return 1e6

    t0 = time.time()
    best_ls, best_sigma = 1.0, 0.1
    best_lml = log_marginal([1.0, 0.1])
    try:
        res = minimize(log_marginal, [1.0, 0.1], method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3})
        if res.fun < best_lml:
            best_lml = float(res.fun)
            best_ls, best_sigma = float(res.x[0]), float(res.x[1])
    except Exception:
        pass

    K = rbf(Xn, Xn, best_ls) + best_sigma ** 2 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
    except Exception:
        return None, None

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        Xqn = (Xq - x_mean) / x_std
        Ks = rbf(Xqn, Xn, best_ls)
        out = (Ks @ alpha) * (y.std() + 1e-8) + y_mean
        return np.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)
    return predict, best_lml


def fit(X_train, y_train):
    global _TOTAL_DEADLINE
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    n, d = X_train.shape

    # Per-call deadline: leave margin for other equations
    call_deadline = time.time() + 70.0  # ~4 * 70 = 280s, safe under 330s

    candidates = []  # list of (nrmse, predict_fn)

    def try_model(name, feat_fn, max_deg=None, max_freq=None):
        if time.time() > call_deadline:
            return
        try:
            if max_deg is not None:
                F = feat_fn(X_train, max_deg=max_deg)
            elif max_freq is not None:
                F = feat_fn(X_train, max_freq=max_freq)
            else:
                F = feat_fn(X_train)
            if not np.all(np.isfinite(F)):
                F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
            w = _fit_linear(F, y_train)
            y_pred = F @ w
            if not np.all(np.isfinite(y_pred)):
                return
            score = _nrmse(y_pred, y_train)
            candidates.append((score, _safe_predict(feat_fn, w)))
        except Exception:
            pass

    # Polynomial degrees
    for deg in [2, 3, 4, 5, 6, 8]:
        try_model(f"poly{deg}", _poly_features, max_deg=deg)

    # Trig frequencies
    for freq in [3, 5, 8]:
        try_model(f"trig{freq}", _trig_features, max_freq=freq)

    # Exponential
    try_model("exp", _exp_features)

    # Mixed
    try_model("mixed", _mixed_features)

    # Spline (only for 1D)
    if d == 1:
        if time.time() < call_deadline:
            try:
                sp_pred = _spline_fit(X_train, y_train, k=5)
                if sp_pred is not None:
                    y_pred = sp_pred(X_train)
                    if np.all(np.isfinite(y_pred)):
                        score = _nrmse(y_pred, y_train)
                        candidates.append((score, sp_pred))
            except Exception:
                pass

        # Higher-order spline
        if time.time() < call_deadline:
            try:
                sp_pred = _spline_fit(X_train, y_train, k=8)
                if sp_pred is not None:
                    y_pred = sp_pred(X_train)
                    if np.all(np.isfinite(y_pred)):
                        score = _nrmse(y_pred, y_train)
                        candidates.append((score, sp_pred))
            except Exception:
                pass

    # Neural net
    if time.time() < call_deadline - 12:
        try:
            nn_pred, _ = _fit_nn(X_train, y_train, time_budget=12.0)
            if nn_pred is not None:
                y_pred = nn_pred(X_train)
                if np.all(np.isfinite(y_pred)):
                    score = _nrmse(y_pred, y_train)
                    candidates.append((score, nn_pred))
        except Exception:
            pass

    # Gaussian process
    if time.time() < call_deadline - 5:
        try:
            gp_pred, _ = _gaussian_process_fit(X_train, y_train, time_budget=8.0)
            if gp_pred is not None:
                y_pred = gp_pred(X_train)
                if np.all(np.isfinite(y_pred)):
                    score = _nrmse(y_pred, y_train)
                    candidates.append((score, gp_pred))
        except Exception:
            pass

    if not candidates:
        # Fallback: constant
        c = float(np.mean(y_train))
        return lambda X: np.full(np.asarray(X).shape[0], c)

    # Pick best by training NRMSE
    candidates.sort(key=lambda t: t[0])
    best_score, best_predict = candidates[0]
    return best_predict
# EVOLVE-BLOCK-END
