"""Tikhonov-regularized degree-5 polynomial fit with Chebyshev basis and SVD solver."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-5 polynomial to the unresolved tendency.
    
    Uses Chebyshev polynomial basis for better numerical conditioning,
    with Tikhonov (L2) regularization in Chebyshev coefficient space.
    Lambda selected by internal k-fold cross-validation.
    """
    t0 = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)

    if n < 6:
        return np.zeros(6)

    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        c = np.zeros(6)
        c[0] = y.mean()
        return c

    # Scale x to [-1, 1] for Chebyshev
    x_scaled = 2.0 * (x - x_min) / x_range - 1.0

    # Chebyshev polynomial basis T_0..T_5
    V = np.zeros((n, 6))
    V[:, 0] = 1.0
    V[:, 1] = x_scaled
    for k in range(2, 6):
        V[:, k] = 2.0 * x_scaled * V[:, k-1] - V[:, k-2]

    # Precompute transformation: Chebyshev coeffs -> raw power coeffs
    a = 2.0 / x_range
    b = -(2.0 * x_min / x_range + 1.0)

    # T_k(t) in powers of t
    T_cheb = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [-1, 0, 2, 0, 0, 0],
        [0, -3, 0, 4, 0, 0],
        [1, 0, -8, 0, 8, 0],
        [0, 5, 0, -20, 0, 16]
    ])

    # Transform from powers of x_scaled to powers of x
    T_pow = np.zeros((6, 6))
    for j in range(6):
        for m in range(j + 1):
            T_pow[m, j] = comb(j, m) * (a ** m) * (b ** (j - m))

    # Full transform: c_raw = T_pow @ T_cheb.T @ c_cheb
    T = T_pow @ T_cheb.T

    # Cross-validation setup
    n_folds = 10
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    # Precompute V^T V and V^T y for each training fold
    # This avoids repeated matrix multiplications during lambda search
    fold_data = []
    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        
        V_train = V[train_idx]
        y_train = y[train_idx]
        V_val = V[val_idx]
        y_val = y[val_idx]
        
        # Precompute Gram matrix and cross-product
        VtV = V_train.T @ V_train  # 6x6
        Vty = V_train.T @ y_train  # 6x1
        
        fold_data.append((VtV, Vty, V_val, y_val))

    def solve_ridge(VtV, Vty, lam):
        """Solve ridge regression in Chebyshev space."""
        if lam == 0.0:
            A = VtV + 1e-14 * np.eye(6)
        else:
            A = VtV + lam * np.eye(6)
        try:
            return np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, Vty, rcond=None)[0]

    def compute_cv_score(lam):
        """Compute mean CV MSE for a given lambda."""
        cv_errors = []
        for VtV, Vty, V_val, y_val in fold_data:
            c_train = solve_ridge(VtV, Vty, lam)
            pred_val = V_val @ c_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))
        if cv_errors:
            return np.mean(cv_errors)
        return np.inf

    # Coarse lambda search - expanded range
    lambdas_coarse = np.concatenate([
        [0.0],
        np.logspace(-12, 8, 60)
    ])

    best_lambda = 0.0
    best_cv_score = np.inf

    for lam in lambdas_coarse:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Fine lambda search around best_lambda
    if best_lambda > 0:
        lower = max(best_lambda * 0.3, 1e-15)
        upper = best_lambda * 3.0
        lambdas_fine = np.logspace(np.log10(lower), np.log10(upper), 30)
    else:
        lambdas_fine = np.logspace(-12, -1, 30)

    for lam in lambdas_fine:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Final fit on all data
    VtV_full = V.T @ V
    Vty_full = V.T @ y

    if best_lambda < 1e-10:
        A = VtV_full + 1e-14 * np.eye(6)
    else:
        A = VtV_full + best_lambda * np.eye(6)

    try:
        coeff_cheb = np.linalg.solve(A, Vty_full)
    except np.linalg.LinAlgError:
        coeff_cheb = np.linalg.lstsq(A, Vty_full, rcond=None)[0]

    # Transform to raw power coefficients
    coeff = T @ coeff_cheb

    if not np.all(np.isfinite(coeff)):
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff
# EVOLVE-BLOCK-END
