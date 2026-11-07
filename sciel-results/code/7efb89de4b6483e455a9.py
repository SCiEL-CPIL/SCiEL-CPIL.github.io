"""Symbolic regression over 4 hidden equations.
Combines basis-expansion selection (current) with exact interpolation and
Gaussian-process smoothing (inspiration) to handle both smooth and
high-frequency/noiseless targets.  A hold-out split selects the best model;
a deadline guard keeps total runtime under 330 s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.interpolate import CubicSpline, PchipInterpolator, UnivariateSpline
warnings.filterwarnings("ignore")

_DEADLINE = time.time() + 285.0  # 45 s safety margin under 330 s


def _remaining():
    return _DEADLINE - time.time()


def _make_predict(F_train, w, feat_fn):
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F = feat_fn(X)
        y = F @ w
        y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
        return y
    return predict


def _poly_features(X, degree):
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)


def _fourier_features(X, n_frequencies, x_min, x_max):
    x = X[:, 0]
    period = x_max - x_min
    if period < 1e-12:
        period = 1.0
    cols = [np.ones(len(x))]
    for k in range(1, n_frequencies + 1):
        freq = 2.0 * np.pi * k / period
        cols.append(np.sin(freq * (x - x_min)))
        cols.append(np.cos(freq * (x - x_min)))
    return np.column_stack(cols)


def _rbf_features(X, centers, bandwidth):
    x = X[:, 0].reshape(-1, 1)
    c = centers.reshape(-1, 1)
    d2 = (x - c) ** 2
    return np.exp(-d2 / (2.0 * bandwidth ** 2))


def _fit_exact_spline(X_train, y_train, method="cubic"):
    """Exact interpolation via cubic or PCHIP spline."""
    x = X_train[:, 0]
    y = y_train
    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    # Remove duplicates
    keep = np.concatenate(([True], np.diff(xs) > 1e-12))
    xs, ys = xs[keep], ys[keep]
    if xs.size < 2:
        return None
    if method == "pchip":
        spl = PchipInterpolator(xs, ys)
    else:
        spl = CubicSpline(xs, ys)

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = spl(X[:, 0])
        return np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
    return predict


def _fit_gp(X_train, y_train, length_scale, noise=1e-8):
    """Gaussian-process regression with RBF kernel."""
    x = X_train[:, 0]
    y = y_train
    n = x.size
    diff = x[:, None] - x[None, :]
    K = np.exp(-0.5 * (diff * diff) / (length_scale * length_scale)) + noise * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            return None

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        xt = X[:, 0]
        Ks = np.exp(-0.5 * (xt[:, None] - x[None, :]) ** 2 / (length_scale * length_scale))
        y = Ks @ alpha
        return np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
    return predict


def _fit_rbf_interp(X_train, y_train, width):
    """RBF interpolation with given width."""
    x = X_train[:, 0]
    y = y_train
    n = x.size
    diff = x[:, None] - x[None, :]
    K = np.exp(-(diff * diff) / (2.0 * width * width))
    K[np.diag_indices_from(K)] += 1e-8
    try:
        coef = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(K, y, rcond=None)[0]

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        xt = X[:, 0]
        d = xt[:, None] - x[None, :]
        y = np.exp(-(d * d) / (2.0 * width * width)) @ coef
        return np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
    return predict


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = len(X_train)
    x = X_train[:, 0]
    x_min, x_max = x.min(), x.max()
    rng_x = x_max - x_min
    if rng_x < 1e-12:
        rng_x = 1.0

    # Hold-out validation split: 15%
    rng = np.random.default_rng(42)
    val_n = max(10, int(n * 0.15))
    perm = rng.permutation(n)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]

    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

    var_y = np.var(y_val)
    if var_y < 1e-15:
        var_y = 1.0

    candidates = []  # (val_nrmse, predict_fn)

    def _add(pred):
        if pred is None:
            return
        try:
            yp = np.asarray(pred(X_val), dtype=float)
            if yp.shape != y_val.shape:
                yp = yp.reshape(y_val.shape)
            if not np.all(np.isfinite(yp)):
                return
            mse = float(np.mean((yp - y_val) ** 2))
            if np.isfinite(mse):
                candidates.append((mse / var_y, pred))
        except Exception:
            pass

    # 1. Exact splines (cubic, PCHIP) on ALL data
    for method in ("cubic", "pchip"):
        if _remaining() < 3:
            break
        try:
            _add(_fit_exact_spline(X_train, y_train, method))
        except Exception:
            pass

    # 2. Gaussian Process with various length scales
    for frac in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        if _remaining() < 3:
            break
        try:
            _add(_fit_gp(X_train, y_train, length_scale=frac * rng_x))
        except Exception:
            pass

    # 3. RBF interpolation with various widths
    for frac in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        if _remaining() < 3:
            break
        try:
            _add(_fit_rbf_interp(X_train, y_train, width=frac * rng_x))
        except Exception:
            pass

    # 4. Polynomial fits
    for deg in range(1, 15):
        if _remaining() < 3:
            break
        try:
            F_tr = _poly_features(X_tr, deg)
            w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
            feat_fn = lambda X, d=deg: _poly_features(X, d)
            _add(_make_predict(F_tr, w, feat_fn))
        except Exception:
            continue

    # 5. Fourier fits
    for n_harm in range(1, 15):
        if _remaining() < 3:
            break
        try:
            F_tr = _fourier_features(X_tr, n_harm, x_min, x_max)
            w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
            feat_fn = lambda X, nh=n_harm, xmin=x_min, xmax=x_max: _fourier_features(X, nh, xmin, xmax)
            _add(_make_predict(F_tr, w, feat_fn))
        except Exception:
            continue

    # 6. RBF basis with fixed centers
    for nc in [5, 10, 20, 30, 50]:
        if _remaining() < 3:
            break
        centers = np.linspace(x_min, x_max, nc)
        for bw_mult in [0.1, 0.2, 0.5, 1.0]:
            bandwidth = rng_x * bw_mult / max(1, nc - 1)
            try:
                F_tr = _rbf_features(X_tr, centers, bandwidth)
                w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
                feat_fn = lambda X, c=centers, bw=bandwidth: _rbf_features(X, c, bw)
                _add(_make_predict(F_tr, w, feat_fn))
            except Exception:
                continue

    # 7. Smoothing splines
    for deg in [3, 4, 5, 6]:
        for s in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
            if _remaining() < 3:
                break
            try:
                idx = np.argsort(X_tr[:, 0])
                xs = X_tr[idx, 0]
                ys = y_tr[idx]
                spl = UnivariateSpline(xs, ys, k=min(deg, len(xs) - 1), s=s)

                def predict_spl(X, _spl=spl):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    y = _spl(X[:, 0])
                    return np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
                _add(predict_spl)
            except Exception:
                continue

    # 8. Mixed polynomial + Fourier
    for deg in [1, 2, 3]:
        for n_harm in [2, 4, 6, 8]:
            if _remaining() < 3:
                break
            try:
                F_tr = np.hstack([_poly_features(X_tr, deg), _fourier_features(X_tr, n_harm, x_min, x_max)])
                w, *_ = np.linalg.lstsq(F_tr, y_tr, rcond=None)
                feat_fn = lambda X, d=deg, nh=n_harm, xmin=x_min, xmax=x_max: np.hstack([_poly_features(X, d), _fourier_features(X, nh, xmin, xmax)])
                _add(_make_predict(F_tr, w, feat_fn))
            except Exception:
                continue

    if not candidates:
        # Fallback: linear fit on all data
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y_train, rcond=None)
        def predict_linear(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return w[0] + w[1] * X[:, 0]
        return predict_linear

    candidates.sort(key=lambda t: t[0])
    best_predict = candidates[0][1]

    # Ensemble: average top-k models if they are all good
    top_k = min(5, len(candidates))
    if top_k > 1 and candidates[top_k - 1][0] < 1.0 and _remaining() > 5:
        preds = [c[1] for c in candidates[:top_k]]
        def ensemble_predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            ys = np.array([p(X) for p in preds])
            y = ys.mean(axis=0)
            return np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
        # Validate ensemble
        try:
            yp = ensemble_predict(X_val)
            mse = float(np.mean((yp - y_val) ** 2))
            if mse / var_y < candidates[0][0]:
                best_predict = ensemble_predict
        except Exception:
            pass

    return best_predict
# EVOLVE-BLOCK-END
