# EVOLVE-BLOCK-START
import time
import numpy as np


def _poly_features(X, d):
    n, p = X.shape
    cols = [np.ones((n, 1))]
    for i in range(p):
        cols.append(X[:, i:i+1])
    for i in range(p):
        for j in range(i, p):
            cols.append((X[:, i] * X[:, j])[:, None])
    if d >= 3:
        for i in range(p):
            cols.append((X[:, i]**3)[:, None])
            for j in range(i, p):
                cols.append((X[:, i]**2 * X[:, j])[:, None])
        for i in range(p):
            for j in range(i, p):
                for k in range(j, p):
                    cols.append((X[:, i] * X[:, j] * X[:, k])[:, None])
    if d >= 4:
        for i in range(p):
            cols.append((X[:, i]**4)[:, None])
            for j in range(i, p):
                cols.append((X[:, i]**3 * X[:, j])[:, None])
                cols.append((X[:, i]**2 * X[:, j]**2)[:, None])
    return np.column_stack(cols)


def _rbf_features(X, centers, bw):
    # X: (n, p), centers: (m, p)
    D2 = (X**2).sum(1)[:, None] - 2.0 * (X @ centers.T) + (centers**2).sum(1)[None, :]
    return np.exp(-D2 / (2.0 * bw * bw))


def _fourier_features(X, freqs):
    cols = [np.ones((X.shape[0], 1))]
    for i in range(X.shape[1]):
        for f in freqs:
            cols.append(np.sin(f * X[:, i])[:, None])
            cols.append(np.cos(f * X[:, i])[:, None])
    return np.column_stack(cols)


def _ridge_solve(A, y, lam):
    n, m = A.shape
    try:
        w = np.linalg.solve(A.T @ A + lam * np.eye(m), A.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, y, rcond=None)[0]
    return w


def _fit_poly(Xtr, ytr, Xva, yva, d, lam=1e-3):
    try:
        Atr = _poly_features(Xtr, d)
        w = _ridge_solve(Atr, ytr, lam)
        pred_va = _poly_features(Xva, d) @ w
        return pred_va, (lambda X: _poly_features(X, d) @ w)
    except Exception:
        return None, None


def _fit_rbf(Xtr, ytr, Xva, yva, bw, lam=1e-2):
    try:
        # Use a subset of training points as centers to keep it fast
        m = min(len(Xtr), 40)
        idx = np.linspace(0, len(Xtr)-1, m, dtype=int)
        centers = Xtr[idx]
        Atr = _rbf_features(Xtr, centers, bw)
        w = _ridge_solve(Atr, ytr, lam)
        pred_va = _rbf_features(Xva, centers, bw) @ w
        return pred_va, (lambda X: _rbf_features(X, centers, bw) @ w)
    except Exception:
        return None, None


def _fit_fourier(Xtr, ytr, Xva, yva, freqs, lam=1e-3):
    try:
        Atr = _fourier_features(Xtr, freqs)
        w = _ridge_solve(Atr, ytr, lam)
        pred_va = _fourier_features(Xva, freqs) @ w
        return pred_va, (lambda X: _fourier_features(X, freqs) @ w)
    except Exception:
        return None, None


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    n, p = X.shape

    # Small validation split for model selection
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_va = max(1, int(0.15 * n))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    best_pred_va = None
    best_pred_fn = None
    best_score = np.inf

    def consider(pred_va, pred_fn):
        nonlocal best_pred_va, best_pred_fn, best_score
        if pred_va is None or pred_fn is None:
            return
        s = np.mean((pred_va - yva) ** 2)
        if s < best_score:
            best_score = s
            best_pred_va = pred_va
            best_pred_fn = pred_fn

    # --- Polynomial models (degrees 2, 3, 4) ---
    for d in (2, 3, 4):
        if time.time() > deadline - 5:
            break
        pv, pf = _fit_poly(Xtr, ytr, Xva, yva, d)
        consider(pv, pf)

    # --- RBF models (several bandwidths) ---
    # Estimate a scale from the data
    spread = np.max(X, axis=0) - np.min(X, axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    typical = float(np.mean(spread))
    for bw in (0.2 * typical, 0.5 * typical, 1.0 * typical, 2.0 * typical):
        if time.time() > deadline - 5:
            break
        pv, pf = _fit_rbf(Xtr, ytr, Xva, yva, bw)
        consider(pv, pf)

    # --- Fourier model ---
    if time.time() < deadline - 5:
        freqs = [0.5, 1.0, 2.0, 4.0]
        pv, pf = _fit_fourier(Xtr, ytr, Xva, yva, freqs)
        consider(pv, pf)

    # Final: refit the chosen family on ALL training data for the returned predictor
    if best_pred_fn is None:
        # Fallback: degree-2 polynomial on all data
        A = _poly_features(X, 2)
        w = _ridge_solve(A, y, 1e-3)
        return lambda X: _poly_features(np.asarray(X, dtype=float), 2) @ w

    # Determine which family won by re-deriving: simplest is to just use the
    # validation-selected predictor but refit on full data for that family.
    # We stored a closure; to refit, detect family by re-fitting all on full data
    # is wasteful. Instead, we refit the winning family on all data.
    # Track which family by re-running selection logic on full data is complex;
    # so we simply refit each family on full data and pick the one whose
    # validation score was best (we have best_score per family already).
    # Simpler & safe: just return the validation-selected closure (it's already
    # a valid predictor trained on ~85% of data). Good enough and safe.
    return lambda X: best_pred_fn(np.asarray(X, dtype=float))
# EVOLVE-BLOCK-END
