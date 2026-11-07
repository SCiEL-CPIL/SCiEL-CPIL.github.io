# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 330.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _ridge_solve(A, b, lam):
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s2 = s ** 2
        denom = s2 + lam
        w = Vt.T @ ((U.T @ b) / denom)
        return w
    except Exception:
        try:
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
            return w
        except Exception:
            return np.zeros(A.shape[1])

def _raw_feature_funcs():
    """Return list of callables f(x)->vector for 1D x."""
    funcs = []
    funcs.append(lambda x: np.ones_like(x))
    for d in range(1, 11):
        funcs.append(lambda x, d=d: x ** d)
    funcs.append(lambda x: np.abs(x))
    funcs.append(np.sin)
    funcs.append(np.cos)
    funcs.append(lambda x: np.sin(2 * x))
    funcs.append(lambda x: np.cos(2 * x))
    funcs.append(np.tanh)
    funcs.append(np.arctan)
    funcs.append(lambda x: np.exp(np.clip(x, -50, 50)))
    funcs.append(lambda x: np.exp(-np.clip(x, -50, 50)))
    funcs.append(lambda x: np.exp(0.5 * np.clip(x, -50, 50)))
    funcs.append(lambda x: np.exp(-0.5 * np.clip(x, -50, 50)))
    for scale in [0.5, 2.0]:
        for d in range(1, 6):
            funcs.append(lambda x, s=scale, d=d: (x * s) ** d)
        funcs.append(lambda x, s=scale: np.sin(x * s))
        funcs.append(lambda x, s=scale: np.cos(x * s))
    funcs.append(lambda x: np.log(np.abs(x) + 1.0))
    funcs.append(lambda x: np.sqrt(np.abs(x) + 1.0))
    return funcs

def _build_raw(x, funcs):
    x = np.asarray(x, dtype=float).ravel()
    cols = []
    for f in funcs:
        try:
            v = f(x)
            v = np.asarray(v, dtype=float).ravel()
            if v.shape[0] != x.shape[0]:
                v = np.full(x.shape[0], np.nan)
            if not np.all(np.isfinite(v)):
                v = np.where(np.isfinite(v), v, 0.0)
            cols.append(v)
        except Exception:
            cols.append(np.zeros_like(x))
    L = np.column_stack(cols)
    # Standardize each column (keep constant column)
    out = np.empty_like(L)
    for j in range(L.shape[1]):
        col = L[:, j]
        std = col.std()
        if std < 1e-12:
            out[:, j] = 1.0 if abs(col.mean()) > 0.5 else 0.0
        else:
            out[:, j] = (col - col.mean()) / std
    return out

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    x = X[:, 0]

    funcs = _raw_feature_funcs()
    L = _build_raw(x, funcs)
    nf = L.shape[1]

    # Drop zero columns
    nonzero = [j for j in range(nf) if np.any(L[:, j] != 0)]
    if not nonzero:
        nonzero = [0]

    # Train/val split
    rng = np.random.default_rng(98765)
    perm = rng.permutation(n)
    nval = max(1, int(0.2 * n))
    val_idx = perm[:nval]
    tr_idx = perm[nval:]

    lam = 1e-3

    chosen = []
    available = list(nonzero)
    best_val = np.inf
    best_chosen = None

    max_features = min(len(available), 25)
    for step in range(max_features):
        if _time_left() < 5.0:
            break
        best_add = None
        best_tr = np.inf
        for c in available:
            cand = chosen + [c]
            A = L[tr_idx, cand]
            w = _ridge_solve(A, y[tr_idx], lam)
            tr_mse = float(np.mean((A @ w - y[tr_idx]) ** 2))
            if tr_mse < best_tr:
                best_tr = tr_mse
                best_add = c
        if best_add is None:
            break
        chosen.append(best_add)
        available.remove(best_add)

        A = L[tr_idx, chosen]
        w = _ridge_solve(A, y[tr_idx], lam)
        val_mse = float(np.mean((L[val_idx, chosen] @ w - y[val_idx]) ** 2))
        if val_mse < best_val:
            best_val = val_mse
            best_chosen = list(chosen)
        if val_mse > best_val * 1.5 and len(chosen) > 3:
            break

    if best_chosen is None:
        best_chosen = [0]
        if nf > 1:
            best_chosen.append(1)

    # Refit on all data
    A = L[:, best_chosen]
    w = _ridge_solve(A, y, lam)
    if not np.all(np.isfinite(w)):
        # fallback linear
        A = np.column_stack([np.ones(n), x])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            best_chosen = [0]  # placeholder; we'll use linear in predict
            use_linear = True
        except Exception:
            w = np.array([float(np.mean(y)), 0.0])
            use_linear = True
    else:
        use_linear = False

    def predict(Xt):
        Xt = np.asarray(Xt, dtype=float)
        if Xt.ndim == 1:
            Xt = Xt.reshape(-1, 1)
        xt = Xt[:, 0]
        if use_linear:
            Llin = np.column_stack([np.ones(Xt.shape[0]), xt])
            return Llin @ w
        Lt = _build_raw(xt, funcs)
        pred = Lt[:, best_chosen] @ w
        return np.where(np.isfinite(pred), pred, float(np.mean(y)))

    return predict
# EVOLVE-BLOCK-END
