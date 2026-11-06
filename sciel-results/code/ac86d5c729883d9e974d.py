"""Tikhonov-regularized degree-five polynomial closure with CV-selected lambda."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _fit_regularized(V, y, lam):
    """Solve (V^T V + lam*I) c = V^T y for polynomial coefficients."""
    n = V.shape[0]
    A = V.T @ V + lam * np.eye(V.shape[1])
    b = V.T @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c


def _predict(V, c):
    return V @ c


def construct(training_x, training_subgrid_tendency):
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(training_x)
    if n < 10:
        # Fallback: just fit a linear polynomial
        if n < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        return coeff

    # Build Vandermonde matrix for degree 5: columns x^0, x^1, ..., x^5
    powers = np.arange(6)
    V = np.power(training_x[:, None], powers[None, :])

    # Normalize x to improve conditioning (standardize)
    x_mean = np.mean(training_x)
    x_std = np.std(training_x)
    if x_std < 1e-12:
        x_norm = np.zeros_like(training_x)
    else:
        x_norm = (training_x - x_mean) / x_std

    V_norm = np.power(x_norm[:, None], powers[None, :])

    # K-fold cross-validation to select lambda
    n_folds = 5
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    fold_sizes = np.array_split(indices, n_folds)

    # Candidate lambdas: log-spaced from small to large
    # Scale lambda relative to the data magnitude for stability
    y_scale = np.std(training_subgrid_tendency)
    if y_scale < 1e-12:
        y_scale = 1.0
    lam_base = (y_scale * x_std ** 5) ** 2  # rough scale
    if lam_base < 1e-12:
        lam_base = 1.0

    lambda_candidates = np.concatenate([
        [0.0],
        np.logspace(-6, 4, 30) * lam_base
    ])

    best_lam = 0.0
    best_cv_score = np.inf

    for lam in lambda_candidates:
        if _time_remaining() < 2.0:
            break
        mse_sum = 0.0
        for fold in range(n_folds):
            val_idx = fold_sizes[fold]
            train_idx = np.concatenate([fold_sizes[i] for i in range(n_folds) if i != fold])

            V_train = V_norm[train_idx]
            y_train = training_subgrid_tendency[train_idx]
            V_val = V_norm[val_idx]
            y_val = training_subgrid_tendency[val_idx]

            c = _fit_regularized(V_train, y_train, lam)
            pred = _predict(V_val, c)
            mse_sum += np.mean((pred - y_val) ** 2)

        cv_score = mse_sum / n_folds
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lam = lam

    # Final fit on all training data with best lambda
    coeff_norm = _fit_regularized(V_norm, training_subgrid_tendency, best_lam)

    # Transform coefficients from normalized x to original x
    # If x_norm = (x - mu)/sigma, then x = sigma * x_norm + mu
    # We need to express sum(c_i * x_norm^i) as sum(d_j * x^j)
    # x_norm^i = ((x - mu)/sigma)^i = sum_{j=0}^{i} binom(i,j) * (x/sigma)^j * (-mu/sigma)^(i-j)
    # Actually: (x - mu)^i / sigma^i = sum_{j=0}^{i} C(i,j) x^j (-mu)^{i-j} / sigma^i
    # So x_norm^i = sum_{j=0}^{i} C(i,j) * (-mu/sigma)^{i-j} * x^j / sigma^0 ... let me redo
    # x_norm = (x - mu)/sigma
    # x_norm^i = (x - mu)^i / sigma^i
    # (x - mu)^i = sum_{j=0}^{i} C(i,j) x^j (-mu)^(i-j)
    # So x_norm^i = sum_{j=0}^{i} C(i,j) (-mu)^(i-j) x^j / sigma^i
    # Therefore: sum_{i=0}^{5} c_i x_norm^i = sum_{j=0}^{5} [sum_{i=j}^{5} c_i * C(i,j) * (-mu)^(i-j) / sigma^i] x^j

    if x_std < 1e-12:
        # x is constant, just return the constant term
        coeff = np.zeros(6)
        coeff[0] = np.mean(training_subgrid_tendency)
        return coeff

    mu = x_mean
    sig = x_std
    coeff = np.zeros(6)
    for j in range(6):
        s = 0.0
        for i in range(j, 6):
            # C(i, j) = i! / (j! (i-j)!)
            from math import comb
            binom_ij = comb(i, j)
            s += coeff_norm[i] * binom_ij * ((-mu) ** (i - j)) / (sig ** i)
        coeff[j] = s

    # Sanity check: ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)[::-1]
        if not np.all(np.isfinite(coeff)):
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
