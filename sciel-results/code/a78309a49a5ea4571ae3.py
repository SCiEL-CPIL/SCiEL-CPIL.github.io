"""Local adaptive polynomial regression (LOESS-like) with LOO-CV bandwidth."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_BUDGET = 280.0  # safety margin under the 330s wall-clock limit


def _local_quad_fit(X, y, Xq, bw):
    """Weighted local quadratic regression at query points Xq using bandwidth bw."""
    d = X.shape[1]
    n = X.shape[0]
    m = Xq.shape[0]
    # basis: 1, x, x^2  (d(d+3)/2 terms)
    bdim = 1 + d + d * (d + 1) // 2
    out = np.empty(m, dtype=float)
    for i in range(m):
        q = Xq[i]
        dist2 = np.sum((X - q) ** 2, axis=1)
        w = np.exp(-0.5 * dist2 / (bw * bw))
        W = np.sqrt(w)
        cols = [np.ones(n)]
        for j in range(d):
            cols.append(X[:, j] * W)
        for j in range(d):
            for k in range(j, d):
                cols.append(X[:, j] * X[:, k] * W)
        B = np.column_stack(cols)
        By = y * W
        # ridge for stability
        lam = 1e-6 * max(1.0, np.trace(B.T @ B) / bdim)
        try:
            coef = np.linalg.solve(B.T @ B + lam * np.eye(bdim), B.T @ By)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(B, By, rcond=None)[0]
        # evaluate basis at q
        basis_q = [1.0]
        for j in range(d):
            basis_q.append(q[j])
        for j in range(d):
            for k in range(j, d):
                basis_q.append(q[j] * q[k])
        out[i] = float(np.dot(basis_q, coef))
    return out


def _loocv_local_quad(X, y, bw):
    """Leave-one-out CV error for local quadratic regression (efficient)."""
    n = X.shape[0]
    d = X.shape[1]
    bdim = 1 + d + d * (d + 1) // 2
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    W = np.exp(-0.5 * D2 / (bw * bw))
    eye = np.eye(n)
    S = np.zeros((n, bdim))
    T = np.zeros((n, bdim))
    lam = 1e-6
    for i in range(n):
        Wi = W[i]
        SWi = np.sqrt(Wi)
        cols = [np.ones(n)]
        for j in range(d):
            cols.append(X[:, j] * SWi)
        for j in range(d):
            for k in range(j, d):
                cols.append(X[:, j] * X[:, k] * SWi)
        B = np.column_stack(cols)
        By = y * SWi
        H = B.T @ B + lam * np.eye(bdim)
        try:
            invH = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            invH = np.linalg.pinv(H)
        S[i] = B.T @ (invH @ By)
        T[i] = B.T @ invH
    # h_ii = sum_k T_ik * S_ik
    h_ii = np.sum(T * S, axis=1)
    denom = 1.0 - h_ii
    denom[denom < 1e-8] = 1e-8
    resid = (y - S) / denom
    return float(np.mean(resid ** 2))


def _global_quad(X, y):
    """Global quadratic least-squares fit."""
    d = X.shape[1]
    n = X.shape[0]
    cols = [np.ones(n)]
    for j in range(d):
        cols.append(X[:, j])
    for j in range(d):
        for k in range(j, d):
            cols.append(X[:, j] * X[:, k])
    B = np.column_stack(cols)
    try:
        coef = np.linalg.lstsq(B, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        coef = np.zeros(B.shape[1])
    return coef


def _global_quad_predict(Xq, coef, d):
    m = Xq.shape[0]
    cols = [np.ones(m)]
    for j in range(d):
        cols.append(Xq[:, j])
    for j in range(d):
        for k in range(j, d):
            cols.append(Xq[:, j] * Xq[:, k])
    Bq = np.column_stack(cols)
    return Bq @ coef


def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    y_mean = float(np.mean(y))

    # global quadratic as a guaranteed baseline
    gcoef = _global_quad(X, y)

    # candidate bandwidths (relative to the data spread)
    spread = float(np.std(X))
    if spread < 1e-8:
        spread = 1.0
    bw_candidates = np.array([0.08, 0.15, 0.3, 0.5, 0.8, 1.2, 1.8]) * spread

    best_bw = bw_candidates[len(bw_candidates) // 2]
    best_cv = np.inf
    used_local = False

    # LOO-CV bandwidth selection (fast for n=200)
    for bw in bw_candidates:
        if time.time() - _T0 > _BUDGET:
            break
        try:
            cv = _loocv_local_quad(X, y, bw)
        except Exception:
            cv = np.inf
        if cv < best_cv:
            best_cv = cv
            best_bw = bw
            used_local = True

    if used_local and best_cv < np.inf:
        def predict(Xq):
            Xq = np.asarray(Xq, dtype=float)
            if Xq.ndim == 1:
                Xq = Xq.reshape(1, -1)
            try:
                out = _local_quad_fit(X, y, Xq, best_bw)
            except Exception:
                out = _global_quad_predict(Xq, gcoef, d)
            return np.asarray(out, dtype=float)
    else:
        def predict(Xq):
            Xq = np.asarray(Xq, dtype=float)
            if Xq.ndim == 1:
                Xq = Xq.reshape(1, -1)
            return _global_quad_predict(Xq, gcoef, d)

    return predict
# EVOLVE-BLOCK-END
