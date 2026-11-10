"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _fit_poly(x, y, deg):
    """Fit a polynomial of given degree, return coefficients padded to length 6."""
    try:
        c = np.polynomial.polynomial.polyfit(x, y, deg)
        if not np.all(np.isfinite(c)):
            raise ValueError("Non-finite")
        out = np.zeros(6)
        out[:deg + 1] = c
        return out
    except Exception:
        return None


def _ridge_fit(x, y, deg, lam):
    """
    Ridge-regularized polynomial fit.
    lam=0 gives least-squares; lam>0 penalizes higher-order terms quadratically.
    Returns coefficient vector of length deg+1, or None on failure.
    """
    V = np.polynomial.polynomial.polyvander(x, deg)
    reg_w = np.array([1.0] + [lam * (i + 1) ** 2 for i in range(1, deg + 1)])
    R = np.diag(reg_w)
    try:
        A = V.T @ V + R
        b = V.T @ y
        c = np.linalg.solve(A, b)
        if not np.all(np.isfinite(c)):
            return None
        return c
    except Exception:
        return None


def _ridge_cv(x, y, deg, lam, n_folds=5, seed=42, deadline=None):
    """
    K-fold cross-validation with ridge regularization.
    Returns mean validation MSE.
    """
    n = len(x)
    if n < n_folds * 2:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        c = _ridge_fit(x, y, deg, lam)
        if c is None:
            return np.inf
        pred = np.polynomial.polynomial.polyval(x, c)
        return np.mean((pred - y) ** 2)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0

    for i in range(n_folds):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < deg + 1:
            continue

        x_tr = x[train_idx]
        y_tr = y[train_idx]
        c = _ridge_fit(x_tr, y_tr, deg, lam)
        if c is None:
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def _soft_min_cv(x, y, deg, lam_grid, n_folds=5, seed=42):
    """
    Smooth surrogate for min over lam_grid of CV score.
    Uses log-sum-exp to get a differentiable-ish proxy that still picks the best lam.
    Returns (best_score, best_lam).
    """
    scores = []
    for lam in lam_grid:
        s = _ridge_cv(x, y, deg, lam, n_folds, seed)
        scores.append(s)
    scores = np.array(scores)
    # Find best
    best_idx = int(np.argmin(scores))
    best_score = float(scores[best_idx])
    best_lam = float(lam_grid[best_idx])
    return best_score, best_lam


def _fista_lasso(A, b, lam, max_iter=2000, tol=1e-8):
    """
    FISTA for Lasso: minimize 0.5*||Ax-b||^2 + lam*||c||_1
    """
    p = A.shape[1]
    AtA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.norm(AtA, ord=2)
    if L < 1e-12:
        L = 1.0
    step = 1.0 / L
    c = np.zeros(p)
    c_prev = np.zeros(p)
    t = 1.0
    for _ in range(max_iter):
        t_prev = t
        c_new = c - step * (AtA @ c - Atb)
        c_new = np.sign(c_new) * np.maximum(np.abs(c_new) - step * lam, 0.0)
        if np.linalg.norm(c_new - c) < tol * (1 + np.linalg.norm(c)):
            c = c_new
            break
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_prev ** 2))
        c = c_new + ((t_prev - 1) / t) * (c_new - c_prev)
        c_prev = c_new
    return c


def _lasso_cv(x, y, deg, lam, n_folds=5, seed=42):
    """K-fold CV with L1 regularization. Returns mean validation MSE."""
    n = len(x)
    if n < n_folds * 2:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        V = np.polynomial.polynomial.polyvander(x, deg)
        c = _fista_lasso(V, y, lam, max_iter=1000)
        pred = np.polynomial.polynomial.polyval(x, c)
        return np.mean((pred - y) ** 2)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    total_mse = 0.0
    valid_folds = 0

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        if len(train_idx) < deg + 1:
            continue
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        V = np.polynomial.polynomial.polyvander(x_tr, deg)
        c = _fista_lasso(V, y_tr, lam, max_iter=1500)
        if not np.all(np.isfinite(c)):
            continue
        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def _polish_coefficients(x, y, coeff, deg, max_iter=50, tol=1e-10):
    """
    High-precision polish: iterate ridge with tiny lambda to tighten solution.
    Uses the current coeff as initialization context via re-solve.
    Returns refined coefficient vector of length 6.
    """
    V = np.polynomial.polynomial.polyvander(x, deg)
    # Start with a very small ridge to stabilize, then reduce
    for lam in [1e-6, 1e-8, 1e-10, 0.0]:
        c = _ridge_fit(x, y, deg, lam)
        if c is not None and np.all(np.isfinite(c)):
            coeff = np.zeros(6)
            coeff[:deg + 1] = c
    return coeff


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin: evaluator kills at 150s

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 6:
        return np.zeros(6)

    best_coeff = None
    best_score = np.inf
    best_deg = 1
    best_lam = 0.0

    # Phase 1: Ridge CV over degrees and regularization strengths
    # Focus on lower effective degrees; heavier regularization for higher orders
    deg_range = [0, 1, 2, 3, 4, 5]
    lam_grid = [0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]

    for deg in deg_range:
        if time.time() > deadline - 20.0:
            break
        for lam in lam_grid:
            if time.time() > deadline - 8.0:
                break
            score = _ridge_cv(x, y, deg, lam, n_folds=5, seed=42, deadline=deadline)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_deg = deg
                best_lam = lam
                c = _ridge_fit(x, y, deg, lam)
                if c is not None:
                    best_coeff = np.zeros(6)
                    best_coeff[:deg + 1] = c

    # Phase 2: Lasso for sparse structure (higher degrees)
    if time.time() < deadline - 20.0:
        lasso_lams = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
        for deg in range(2, 6):
            if time.time() > deadline - 8.0:
                break
            for lam in lasso_lams:
                if time.time() > deadline - 4.0:
                    break
                score = _lasso_cv(x, y, deg, lam, n_folds=5, seed=42)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_deg = deg
                    best_lam = lam
                    V = np.polynomial.polynomial.polyvander(x, deg)
                    c = _fista_lasso(V, y, lam, max_iter=3000)
                    if np.all(np.isfinite(c)):
                        best_coeff = np.zeros(6)
                        best_coeff[:deg + 1] = c

    # Phase 3: High-precision polish on the best model
    if best_coeff is not None and time.time() < deadline - 5.0:
        polished = _polish_coefficients(x, y, best_coeff, best_deg)
        if np.all(np.isfinite(polished)):
            # Verify polish didn't hurt (check training MSE)
            pred = np.polynomial.polynomial.polyval(x, polished)
            train_mse = np.mean((pred - y) ** 2)
            pred_old = np.polynomial.polynomial.polyval(x, best_coeff)
            train_mse_old = np.mean((pred_old - y) ** 2)
            if train_mse <= train_mse_old * 1.001:  # allow tiny numerical drift
                best_coeff = polished

    if best_coeff is not None and np.all(np.isfinite(best_coeff)):
        return best_coeff

    # Fallback
    coeff = _fit_poly(x, y, 1)
    if coeff is not None:
        return coeff
    return np.zeros(6)
# EVOLVE-BLOCK-END
