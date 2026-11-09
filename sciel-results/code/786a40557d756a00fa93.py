import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

_START = time.time()
_DEADLINE = _START + 315.0          # safety margin before the 330s kill
_BUDGET_PER_EQUATION = 25.0         # generous per-equation cap (10 equations)


def _fit_gp(X, y, d, max_iter=250):
    """Fit a zero-mean GP with RBF kernel via Max Likelihood (log-likelihood)."""
    n = X.shape[0]

    # standardize inputs (and keep y in original scale)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    def K(Xa, Xb, l, sn):
        # squared distance
        diff = Xa[:, None, :] - Xb[None, :, :]
        sq = np.sum(diff * diff, axis=-1)
        return np.exp(-0.5 * sq / (l * l)) + sn * np.eye(Xa.shape[0])

    y0 = float(np.mean(y))
    yv = y - y0
    yv2 = yv * yv

    def params_to_lsn(p):
        l = float(np.exp(p[0]))
        sn = float(np.exp(p[1]))
        return l, sn

    def negloglik(p):
        l, sn = params_to_lsn(p)
        Kmat = K(Xs, Xs, l, sn)
        try:
            L = np.linalg.cholesky(Kmat)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yv))
        ll = (
            -0.5 * float(yv @ alpha)
            - np.sum(np.log(np.diag(L)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )
        return -ll

    # initial length scale ~ typical nearest-neighbor distance
    nn = np.sqrt(np.sum((Xs - Xs[0]) ** 2, axis=1))
    nn = nn[nn > 1e-9]
    l0 = float(np.median(nn)) if nn.size else 1.0
    l0 = min(max(l0, 0.05), 3.0)

    x0 = np.array([np.log(l0), np.log(1e-4)])
    best_x = x0.copy()
    best_f = negloglik(x0)

    def _time_ok():
        return (time.time() - _START) < _DEADLINE

    t0 = time.time()
    for _ in range(max_iter):
        if not _time_ok() or (time.time() - t0) > _BUDGET_PER_EQUATION:
            break
        res = minimize(negloglik, best_x, method='L-BFGS-B',
                       options={'maxiter': 20, 'maxfun': 200})
        if res.success and res.fun < best_f:
            best_f = res.fun
            best_x = res.x
        else:
            # small nudge to escape a stuck region
            best_x = best_x + np.array([0.1 * np.random.randn(2)])
        if best_f > 1e12:
            break

    l, sn = params_to_lsn(best_x)
    return y0, mu, sd, l, sn


def _fit_fallback(X, y, d):
    """Robust least-squares on a rich feature basis (polynomial + trig + exp/radial)."""
    n = X.shape[0]
    cols = [np.ones(n)]
    for j in range(d):
        cols.append(X[:, j])
    # quadratic
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    # trigonometric (low frequency)
    for j in range(d):
        for f in (1.0, 2.0):
            cols.append(np.sin(f * X[:, j]))
            cols.append(np.cos(f * X[:, j]))
    # radial / exp
    r = np.sqrt(np.sum(X * X, axis=1))
    cols.append(r)
    cols.append(np.exp(-0.5 * r))
    cols.append(1.0 / (1.0 + r))
    F = np.column_stack(cols)
    try:
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
    except np.linalg.LinAlgError:
        w = np.zeros(F.shape[1])
    return F, w


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if X_train.ndim == 1:
        X_train = X_train[:, None]
    d = X_train.shape[1]
    n = X_train.shape[0]

    # Always compute the robust fallback first (cheap, guaranteed finite).
    Fb, wb = _fit_fallback(X_train, y_train, d)

    gp = None
    try:
        if (time.time() - _START) < _DEADLINE:
            gp = _fit_gp(X_train, y_train, d)
    except Exception:
        gp = None

    if gp is not None:
        y0, mu, sd, l, sn = gp

        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X[:, None]
            Xs = (X - mu) / sd
            try:
                Kmat = np.exp(-0.5 * np.sum((Xs - Xs[0]) ** 2, axis=1)) + sn * np.eye(Xs.shape[0])
                L = np.linalg.cholesky(Kmat)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train - y0))
                Kst = np.exp(-0.5 * np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=-1) / (l * l))
                out = y0 + Kst @ alpha
                # guard against any non-finite value
                if not np.all(np.isfinite(out)):
                    out = Fb @ wb
                return out
            except Exception:
                return Fb @ wb
    else:
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X[:, None]
            n2 = X.shape[0]
            cols = [np.ones(n2)]
            for j in range(d):
                cols.append(X[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X[:, i] * X[:, j])
            for j in range(d):
                for f in (1.0, 2.0):
                    cols.append(np.sin(f * X[:, j]))
                    cols.append(np.cos(f * X[:, j]))
            r = np.sqrt(np.sum(X * X, axis=1))
            cols.append(r)
            cols.append(np.exp(-0.5 * r))
            cols.append(1.0 / (1.0 + r))
            F = np.column_stack(cols)
            out = F @ wb
            if not np.all(np.isfinite(out)):
                out = np.full(n2, float(np.mean(y_train)))
            return out

    return predict
