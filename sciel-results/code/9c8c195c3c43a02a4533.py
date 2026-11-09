"""Symbolic regression over hidden 1-D equations: scaled basis + ridge, CV-regularized."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_basis(x):
    """Bounded, well-scaled feature basis for a 1-D input x (array)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.zeros((0, 1))
    lo, hi = float(np.min(x)), float(np.max(x))
    rng = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    u = (x - lo) / rng  # in [0,1]
    t = 2.0 * u - 1.0   # in [-1,1]

    cols = [np.ones(n), t]
    # Bounded polynomials in t (up to degree 9)
    for k in range(2, 10):
        cols.append(t ** k)
    # Trig terms (bounded)
    for k in range(1, 6):
        cols.append(np.sin(k * np.pi * t))
        cols.append(np.cos(k * np.pi * t))
    # Bounded exponentials via bounded argument in [-1,1]
    for s in (1.0, -1.0):
        cols.append(np.exp(s * 2.0 * t))
    return np.column_stack(cols)


def _ridge_fit(F, y, lam):
    # Solve (F^T F + lam I) w = F^T y with a tiny diagonal jitter for stability.
    A = F.T @ F + lam * np.eye(F.shape[1])
    A[0, 0] += 1e-12
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w


def _predict(F, w):
    p = F @ w
    return np.where(np.isfinite(p), p, 0.0)


def fit(X_train, y_train):
    start = time.time()
    deadline = start + 260.0  # safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).ravel()
    n = y.size
    d = X.shape[1] if X.ndim == 1 else X.shape[1]
    x = X[:, 0] if X.ndim == 2 else X.ravel()

    F = _build_basis(x)
    # Standardize columns for conditioning (keep intercept).
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Fz = (F - mu) / sd
    Fz[:, 0] = 1.0

    ym = y.mean()
    ys = y.std()
    if ys < 1e-12:
        ys = 1.0
    yz = (y - ym) / ys

    # Cross-validated ridge lambda selection (fast).
    best_lam = 1.0
    best_score = np.inf
    for lam in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        if time.time() > deadline - 30.0:
            break
        k = 5
        idx = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(idx)
        folds = np.array_split(idx, k)
        se = 0.0
        for tr, te in folds:
            w = _ridge_fit(Fz[tr], yz[tr], lam)
            pred = _predict(Fz[te], w)
            se += np.mean((pred - yz[te]) ** 2)
        score = se / k
        if score < best_score:
            best_score = score
            best_lam = lam

    w = _ridge_fit(Fz, yz, best_lam)

    def predict(Xp):
        Xp = np.asarray(Xp, dtype=np.float64)
        xp = Xp[:, 0] if Xp.ndim == 2 else Xp.ravel()
        Fp = _build_basis(xp)
        Fpz = (Fp - mu) / sd
        Fpz[:, 0] = 1.0
        p = _predict(Fpz, w)
        return (p * ys) + ym

    return predict
# EVOLVE-BLOCK-END
