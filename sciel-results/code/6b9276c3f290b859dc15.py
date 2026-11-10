"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _polyval_centered(x, c_centered, mu, sigma):
    """Evaluate polynomial given coefficients in centered/scaled basis."""
    xc = (x - mu) / sigma
    result = np.zeros_like(xc, dtype=float)
    for k in range(len(c_centered) - 1, -1, -1):
        result = result * xc + c_centered[k]
    return result


def _convert_to_original_basis(c_centered, mu, sigma, degree):
    """Convert coefficients from centered/scaled basis to original x basis."""
    a = 1.0 / sigma
    b = -mu / sigma
    coeff_orig = np.zeros(degree + 1)
    for k in range(degree + 1):
        ck = c_centered[k]
        if abs(ck) < 1e-30:
            continue
        for i in range(k + 1):
            # binom(k, i)
            binom = 1.0
            for j in range(i):
                binom *= (k - j) / (j + 1)
            coeff_orig[i] += ck * binom * (a ** i) * (b ** (k - i))
    return coeff_orig


def _fit_ridge_centered(x, y, degree, alpha):
    """Fit degree-`degree` polynomial in centered/scaled x with ridge regularization.
    Returns (coeff_original, mu, sigma)."""
    n = len(x)
    if n == 0:
        return np.zeros(degree + 1), 0.0, 1.0

    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0

    xc = (x - mu) / sigma
    V = np.column_stack([xc ** k for k in range(degree + 1)])

    A = V.T @ V
    b = V.T @ y

    if alpha > 0:
        reg = alpha * np.eye(degree + 1)
        reg[0, 0] += 1e-12
        try:
            c_centered = np.linalg.solve(A + reg, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A + reg, b, rcond=None)
    else:
        try:
            c_centered = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_centered, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    coeff_orig = _convert_to_original_basis(c_centered, mu, sigma, degree)
    return coeff_orig, mu, sigma


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0

    x = np.asarray(training_x, dtype=float).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=float).flatten()

    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return np.zeros(6)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        coeff = np.zeros(6)
        if n >= 2:
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            except Exception:
                coeff[0] = np.mean(y)
        elif n >= 1:
            coeff[0] = np.mean(y)
        return coeff

    # Internal 5-fold CV to select degree and alpha
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // 5
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(5)]
    if len(folds) < 5 or len(folds[-1]) == 0:
        # Fallback: simple fit
        try:
            coeff, _, _ = _fit_ridge_centered(x, y, 3, 1e-4)
            return coeff
        except Exception:
            return np.zeros(6)

    # Candidate (degree, alpha) pairs
    candidates = [
        (5, 1e-4), (5, 1e-3), (5, 1e-2), (5, 0.1), (5, 1.0),
        (4, 1e-4), (4, 1e-3), (4, 1e-2), (4, 0.1), (4, 1.0),
        (3, 1e-4), (3, 1e-3), (3, 1e-2), (3, 0.1), (3, 1.0),
        (2, 1e-3), (2, 1e-2), (2, 0.1), (2, 1.0),
    ]

    best_cv_score = np.inf
    best_params = (3, 1e-3)

    for deg, alpha in candidates:
        if time.time() > deadline - 10.0:
            break
        cv_errors = []
        for i in range(5):
            if time.time() > deadline - 10.0:
                break
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            x_train = x[train_idx]
            y_train = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]

            try:
                coeff_fit, mu_fit, sigma_fit = _fit_ridge_centered(x_train, y_train, deg, alpha)
                y_pred = np.polynomial.polynomial.polyval(x_val, coeff_fit)
                mse = np.mean((y_val - y_pred) ** 2)
                cv_errors.append(mse)
            except Exception:
                cv_errors.append(np.inf)

        if len(cv_errors) < 5:
            continue
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_score:
            best_cv_score = mean_cv_error
            best_params = (deg, alpha)

    deg, alpha = best_params

    # Final fit on all training data with best parameters
    try:
        best_coeff, _, _ = _fit_ridge_centered(x, y, deg, alpha)
    except Exception:
        # Fallback
        try:
            best_coeff = np.polynomial.polynomial.polyfit(x, y, min(deg, 3))
            if len(best_coeff) < 6:
                tmp = np.zeros(6)
                tmp[:len(best_coeff)] = best_coeff
                best_coeff = tmp
        except Exception:
            best_coeff = np.zeros(6)

    # Ensure exactly 6 finite coefficients
    if len(best_coeff) < 6:
        tmp = np.zeros(6)
        tmp[:len(best_coeff)] = best_coeff
        best_coeff = tmp
    best_coeff = best_coeff[:6]
    best_coeff = np.where(np.isfinite(best_coeff), best_coeff, 0.0)

    return best_coeff
# EVOLVE-BLOCK-END
