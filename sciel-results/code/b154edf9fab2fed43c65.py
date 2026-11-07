"""Symbolic regression over hidden equations. Rich basis + forward feature selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 300.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _build_basis(X):
    """Build a rich feature matrix: powers, products, trig, exp, log, tanh."""
    n, d = X.shape
    cols = [np.ones(n, dtype=float)]
    for j in range(d):
        x = X[:, j].astype(float)
        cols.append(x)
        cols.append(x * x)
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(np.exp(-0.3 * x))
    # pairwise products of raw inputs
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    # some pairwise trig products
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
    B = np.column_stack(cols)
    # normalize columns for numerical stability (keep a copy of scale for predict)
    scale = np.where(np.abs(B).max(axis=0) > 0, np.abs(B).max(axis=0), 1.0)
    Bn = B / scale
    return Bn, scale


def _fit_lsq(Bn, y, reg):
    # solve (Bn^T Bn + reg I) w = Bn^T y
    G = Bn.T @ Bn
    G.flat[::Bn.shape[1] + 1] += reg
    rhs = Bn.T @ y
    try:
        w = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Bn, y, rcond=None)[0]
    return w


def _mse(Bn, w, y):
    r = Bn @ w - y
    return float(np.mean(r * r))


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    Bn, scale = _build_basis(X_train)
    p = Bn.shape[1]

    # center y for stability
    ymean = float(np.mean(y_train))
    yc = y_train - ymean
    yvar = float(np.var(y_train)) if y_train.size > 1 else 1.0
    if yvar <= 1e-12:
        yvar = 1e-12

    # forward feature selection
    reg = 1e-6
    selected = [0]  # intercept
    best_mse = float(np.mean(yc * yc))
    avail = list(range(1, p))

    # precompute column norms for cheap candidate scoring
    col_norm = np.sqrt(np.sum(Bn * Bn, axis=0))
    # initial residual
    resid = yc.copy()

    while _time_left() > 5.0 and len(selected) < p:
        # score each candidate by reduction in residual norm using projection
        # r_new = r - (Bn[:,j] . r / (Bn[:,j].Bn[:,j] + reg)) * Bn[:,j]
        best_j = -1
        best_score = best_mse
        # project residual onto selected subspace-free: use orthogonal-ish greedy
        # For speed, approximate using correlation with current residual
        corr = Bn.T @ resid
        denom = col_norm * col_norm + reg
        for j in avail:
            if j in selected:
                continue
            a = corr[j] / denom[j]
            r2 = resid - a * Bn[:, j]
            mse = float(np.mean(r2 * r2))
            if mse < best_score:
                best_score = mse
                best_j = j
        if best_j < 0:
            break
        selected.append(best_j)
        avail.remove(best_j)
        # refit on selected with regularization
        Bs = Bn[:, selected]
        w = _fit_lsq(Bs, yc, reg)
        cur_mse = _mse(Bs, w, yc)
        # early stop if improvement negligible
        if best_mse - cur_mse < 1e-9 * max(1.0, best_mse):
            best_mse = cur_mse
            break
        best_mse = cur_mse
        resid = Bs @ w - yc

    Bs = Bn[:, selected]
    w = _fit_lsq(Bs, yc, reg)

    def predict(X):
        X = np.asarray(X, dtype=float)
        Bn_x, _ = _build_basis(X)
        return float(ymean) + float(Bn_x[:, selected] @ w) if X.shape[0] == 1 else (ymean + Bn_x[:, selected] @ w)

    return predict
# EVOLVE-BLOCK-END
