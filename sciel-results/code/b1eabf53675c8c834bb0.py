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


def _cv_score(x, y, deg, n_folds=5, seed=0):
    """
    Cross-validation MSE for a polynomial of given degree.
    Uses K-fold CV to estimate generalization error.
    """
    n = len(x)
    if n < n_folds * 2:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        # Not enough data for CV; use training error as proxy
        c = _fit_poly(x, y, deg)
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
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        if len(train_idx) < deg + 1:
            continue

        c = _fit_poly(x[train_idx], y[train_idx], deg)
        if c is None:
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def _regularized_cv_score(x, y, deg, n_folds=5, seed=0, reg_lambda=0.0):
    """
    CV score with optional ridge regularization on higher-order terms.
    """
    if reg_lambda <= 0.0:
        return _cv_score(x, y, deg, n_folds, seed)

    n = len(x)
    if n < n_folds * 2:
        n_folds = max(2, n // 2)
    if n_folds < 2:
        # Fallback to unregularized
        return _cv_score(x, y, deg, n_folds, seed)

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

        # Build Vandermonde matrix for training
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        V = np.polynomial.polynomial.polyvander(x_tr, deg)

        # Ridge regularization: penalize higher-order coefficients more
        # This acts as a smooth surrogate for degree selection
        lam = reg_lambda
        # Regularization matrix: weight for each coefficient
        # Heavier penalty on higher powers
        reg_weights = np.array([1.0] + [lam * (i + 1) ** 2 for i in range(1, deg + 1)])
        R = np.diag(reg_weights)

        try:
            # Solve (V^T V + R) c = V^T y
            A = V.T @ V + R
            b = V.T @ y_tr
            c = np.linalg.solve(A, b)
            if not np.all(np.isfinite(c)):
                continue
        except Exception:
            continue

        pred = np.polynomial.polynomial.polyval(x[val_idx], c)
        total_mse += np.mean((pred - y[val_idx]) ** 2)
        valid_folds += 1

    if valid_folds == 0:
        return np.inf
    return total_mse / valid_folds


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Ensure finite inputs
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 6:
        return np.zeros(6)

    # Try degrees 0 through 5, using CV with heavy higher-order regularization
    # to select the effective degree that generalizes best
    best_coeff = None
    best_score = np.inf

    # Regularization strength for higher-order terms
    # Heavy regularization encourages lower effective degree
    reg_lambdas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

    for deg in range(6):
        if time.time() > deadline - 5.0:
            break

        for reg_lam in reg_lambdas:
            if time.time() > deadline - 3.0:
                break

            score = _regularized_cv_score(x, y, deg, n_folds=5, seed=42, reg_lambda=reg_lam)

            if score < best_score:
                best_score = score
                # Refit on all data with the selected degree and regularization
                if reg_lam > 0:
                    V = np.polynomial.polynomial.polyvander(x, deg)
                    reg_weights = np.array([1.0] + [reg_lam * (i + 1) ** 2 for i in range(1, deg + 1)])
                    R = np.diag(reg_weights)
                    try:
                        A = V.T @ V + R
                        b = V.T @ y
                        c = np.linalg.solve(A, b)
                        if np.all(np.isfinite(c)):
                            best_coeff = np.zeros(6)
                            best_coeff[:deg + 1] = c
                    except Exception:
                        best_coeff = _fit_poly(x, y, deg)
                else:
                    best_coeff = _fit_poly(x, y, deg)

    if best_coeff is not None and np.all(np.isfinite(best_coeff)):
        return best_coeff

    # Fallback: linear
    coeff = _fit_poly(x, y, 1)
    if coeff is not None:
        return coeff

    return np.zeros(6)
# EVOLVE-BLOCK-END
