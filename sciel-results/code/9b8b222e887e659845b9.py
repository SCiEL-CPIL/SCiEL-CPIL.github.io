# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, RBFInterpolator
from sklearn.model_selection import KFold

_TIME_LIMIT = 330.0
_DEADLINE = time.time() + _TIME_LIMIT - 20.0  # safety margin

def _as_1d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X
    if X.shape[1] == 1:
        return X[:, 0]
    return X  # caller handles

def _poly_predict(X, w, d):
    X = _as_1d(X)
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0:
        return np.empty(0)
    P = np.vander(X, N=d + 1, increasing=True)
    return P @ w

def _poly_cv(X, y, d, k=10):
    kf = KFold(n_splits=k, shuffle=False)
    sse = 0.0
    for tr, te in kf.split(X):
        Pt = np.vander(X[tr], N=d + 1, increasing=True)
        try:
            w = np.linalg.lstsq(Pt, y[tr], rcond=None)[0]
        except Exception:
            return np.inf
        sse += np.sum((np.vander(X[te], N=d + 1, increasing=True) @ w - y[te]) ** 2)
    return sse

def _pchip_predict(X, p):
    X = _as_1d(X)
    X = np.asarray(X, dtype=float)
    return p(X)

def _pchip_cv(X, y, k=10):
    xs = np.asarray(X, dtype=float)
    ys = np.asarray(y, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # require strictly increasing x for PCHIP
    if np.any(np.diff(xs) <= 0):
        return np.inf
    kf = KFold(n_splits=k, shuffle=False)
    sse = 0.0
    for tr, te in kf.split(xs):
        try:
            p = PchipInterpolator(xs[tr], ys[tr])
            pred = p(xs[te])
        except Exception:
            return np.inf
        if not np.all(np.isfinite(pred)):
            return np.inf
        sse += np.sum((pred - ys[te]) ** 2)
    return sse

def _cs_predict(X, p):
    X = _as_1d(X)
    X = np.asarray(X, dtype=float)
    return p(X)

def _cs_cv(X, y, k=10):
    xs = np.asarray(X, dtype=float)
    ys = np.asarray(y, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.any(np.diff(xs) <= 0):
        return np.inf
    kf = KFold(n_splits=k, shuffle=False)
    sse = 0.0
    for tr, te in kf.split(xs):
        try:
            p = CubicSpline(xs[tr], ys[tr])
            pred = p(xs[te])
        except Exception:
            return np.inf
        if not np.all(np.isfinite(pred)):
            return np.inf
        sse += np.sum((pred - ys[te]) ** 2)
    return sse

def _rbf_predict(X, p):
    X = _as_1d(X)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X2 = X.reshape(-1, 1)
    else:
        X2 = X
    return p(X2)

def _rbf_cv(X, y, k=10):
    xs = np.asarray(X, dtype=float).reshape(-1, 1)
    ys = np.asarray(y, dtype=float)
    kf = KFold(n_splits=k, shuffle=False)
    sse = 0.0
    for tr, te in kf.split(xs):
        try:
            p = RBFInterpolator(xs[tr], ys[tr], kernel='thin_plate_spline')
            pred = p(xs[te])
        except Exception:
            return np.inf
        if not np.all(np.isfinite(pred)):
            return np.inf
        sse += np.sum((pred - ys[te]) ** 2)
    return sse

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d_in = X.shape[1]

    # Build candidate models, each as (cv_score, predict_fn).
    candidates = []

    if d_in == 1:
        # 1-D noiseless -> interpolation is the strongest prior.
        for name, cvfn, mkpred in (
            ('pchip', _pchip_cv, _pchip_predict),
            ('cs', _cs_cv, _cs_predict),
            ('rbf', _rbf_cv, _rbf_predict),
        ):
            try:
                if time.time() > _DEADLINE:
                    break
                sse = cvfn(X, y)
                if np.isfinite(sse):
                    # build full model
                    if name == 'pchip':
                        order = np.argsort(X[:, 0])
                        p = PchipInterpolator(X[order, 0], y[order])
                        candidates.append((sse, p))
                    elif name == 'cs':
                        order = np.argsort(X[:, 0])
                        p = CubicSpline(X[order, 0], y[order])
                        candidates.append((sse, p))
                    else:
                        p = RBFInterpolator(X, y, kernel='thin_plate_spline')
                        candidates.append((sse, p))
            except Exception:
                pass

    # Polynomial fallbacks (always safe, finite).
    for d in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
        try:
            if time.time() > _DEADLINE:
                break
            sse = _poly_cv(X, y, d)
            if not np.isfinite(sse):
                continue
            w = np.linalg.lstsq(np.vander(X[:, 0], N=d + 1, increasing=True), y, rcond=None)[0]
            candidates.append((sse, w))
        except Exception:
            pass

    # Pick the best candidate by CV score.
    if candidates:
        candidates.sort(key=lambda t: t[0])
        best = candidates[0]
        if isinstance(best[1], np.ndarray):
            w = best[1]
            d = w.shape[0] - 1
            return (lambda X: _poly_predict(X, w, d))
        else:
            p = best[1]
            return (lambda X: _pchip_predict(X, p))

    # Absolute fallback: constant mean (guaranteed finite).
    c = float(np.mean(y))
    return (lambda X: np.full(len(np.asarray(X).reshape(-1, 1)), c))
# EVOLVE-BLOCK-END
