"""Rewritten program: Tikhonov-regularized degree-5 polynomial fit with internal CV."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-5 polynomial to the unresolved tendency.
    
    Uses Tikhonov (L2) regularization with the strength selected by internal
    k-fold cross-validation to balance training fit and generalization to
    unseen forcing regimes (e.g., F=10).
    """
    t0 = time.time()
    DEADLINE = 120.0  # seconds; leave margin for evaluator overhead
    SAFETY_MARGIN = 5.0  # return at least 5s before deadline

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)

    if n < 6:
        return np.zeros(6)

    # Build Vandermonde matrix for degree 5: columns are x^0, x^1, ..., x^5
    # Use scaled x for numerical stability
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        # Degenerate case: constant x
        c = np.zeros(6)
        c[0] = y.mean()
        return c

    # Scale x to [-1, 1] for better conditioning
    x_scaled = 2.0 * (x - x_min) / x_range - 1.0

    V = np.vander(x_scaled, N=6, increasing=True)  # n x 6, columns x^0..x^5

    # Regularization strengths to try (log-spaced)
    # Include 0 (no regularization) and a range of lambda values
    lambdas = np.concatenate([
        [0.0],
        np.logspace(-10, 6, 40)
    ])

    n_folds = 5
    # Use stratified-ish split based on x values
    indices = np.arange(n)
    # Shuffle deterministically
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    best_lambda = 0.0
    best_cv_score = np.inf
    best_cv_coeffs = None

    # Precompute fold sizes
    for lam in lambdas:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_errors = []
        for f in range(n_folds):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])

            V_train = V[train_idx]
            y_train = y[train_idx]
            V_val = V[val_idx]
            y_val = y[val_idx]

            # Solve regularized least squares: (V^T V + lam * I) c = V^T y
            # But we should not penalize the intercept as heavily; use a diag penalty
            # For simplicity, uniform L2 penalty
            if lam == 0.0:
                # Unregularized: use lstsq
                try:
                    c_train, *_ = np.linalg.lstsq(V_train, y_train, rcond=None)
                except np.linalg.LinAlgError:
                    continue
            else:
                A = V_train.T @ V_train + lam * np.eye(6)
                b = V_train.T @ y_train
                try:
                    c_train = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue

            # Prediction on validation
            pred_val = V_val @ c_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))

        if cv_errors:
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_cv_score:
                best_cv_score = mean_cv
                best_lambda = lam
                best_cv_coeffs = c_train

    # Final fit on all data with best lambda
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        # Return best found so far; if none, fall back to unregularized
        if best_cv_coeffs is not None:
            # Transform coefficients back from scaled to original x
            return _scale_coeffs(best_cv_coeffs, x_min, x_range)
        else:
            return np.zeros(6)

    if best_lambda == 0.0:
        try:
            coeff_scaled, *_ = np.linalg.lstsq(V, y, rcond=None)
        except np.linalg.LinAlgError:
            coeff_scaled = np.zeros(6)
    else:
        A = V.T @ V + best_lambda * np.eye(6)
        b = V.T @ y
        try:
            coeff_scaled = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            try:
                coeff_scaled, *_ = np.linalg.lstsq(V, y, rcond=None)
            except np.linalg.LinAlgError:
                coeff_scaled = np.zeros(6)

    # Transform coefficients from scaled x to original x
    coeff = _scale_coeffs(coeff_scaled, x_min, x_range)

    # Ensure finite
    if not np.all(np.isfinite(coeff)):
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff


def _scale_coeffs(coeff_scaled, x_min, x_range):
    """Convert polynomial coefficients from x_scaled = 2(x-x_min)/x_range - 1
    back to original x: p(x) = sum c_i * ((2(x-x_min)/x_range - 1)^i)
    
    We need to express p(x) = sum d_j * x^j.
    
    Let t = (2/x_range) * x - (2*x_min/x_range + 1) = a*x + b
    where a = 2/x_range, b = -(2*x_min/x_range + 1)
    
    p(x) = sum_{i=0}^{5} c_i * (a*x + b)^i
    
    Expand (a*x + b)^i = sum_{j=0}^{i} C(i,j) * a^j * b^{i-j} * x^j
    
    So d_j = sum_{i=j}^{5} c_i * C(i,j) * a^j * b^{i-j}
    """
    a = 2.0 / x_range
    b = -(2.0 * x_min / x_range + 1.0)
    
    d = np.zeros(6)
    for i in range(6):
        ci = coeff_scaled[i]
        for j in range(i + 1):
            from math import comb
            d[j] += ci * comb(i, j) * (a ** j) * (b ** (i - j))
    
    return d
# EVOLVE-BLOCK-END
