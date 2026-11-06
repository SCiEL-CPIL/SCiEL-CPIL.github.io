"""Chebyshev-basis Tikhonov-regularized degree-5 polynomial fit with CV."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import comb


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-5 polynomial to the unresolved tendency.
    
    Uses a Chebyshev basis in the scaled domain [-1, 1] for improved numerical
    stability and conditioning. Tikhonov regularization is applied to the
    Chebyshev coefficients, which effectively penalizes high-frequency
    oscillations. The regularization strength is selected via 5-fold
    cross-validation. The final coefficients are converted back to the
    monomial basis in the original domain.
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

    # Scale x to [-1, 1]
    x_scaled = 2.0 * (x - x_min) / x_range - 1.0
    
    # Construct Chebyshev design matrix
    # C[i, j] = T_j(x_scaled[i])
    C = np.zeros((n, 6))
    # T_0(x) = 1
    C[:, 0] = 1.0
    # T_1(x) = x
    C[:, 1] = x_scaled
    # T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
    for k in range(2, 6):
        C[:, k] = 2.0 * x_scaled * C[:, k-1] - C[:, k-2]

    # Precompute transformation matrix from scaled monomial coeffs to original monomial coeffs
    # x_orig = (x_scaled + 1) / 2 * x_range + x_min
    # x_scaled = 2 * (x_orig - x_min) / x_range - 1
    # Let x_s be scaled, x_o be original.
    # x_o = A * x_s + B
    # A = x_range / 2
    # B = x_min + x_range / 2
    # We fit P(x_s) = sum c_s[i] x_s^i.
    # We want P(x_o) = sum c_o[i] x_o^i.
    # P(x_o) = P(A*x_o + B) = sum c_s[i] (A*x_o + B)^i
    # Expand (A*x_o + B)^i using binomial theorem: sum_{j=0}^i comb(i, j) A^j B^(i-j) x_o^j
    # Coefficient of x_o^j in P(x_o) is sum_{i=j}^5 c_s[i] * comb(i, j) * A^j * B^(i-j)
    
    A = x_range / 2.0
    B = x_min + x_range / 2.0
    
    T_trans = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            T_trans[j, i] = comb(i, j) * (A ** j) * (B ** (i - j))
            
    # Cross-validation setup
    n_folds = 5
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    # Precompute fold data
    fold_data = []
    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        
        C_train = C[train_idx]
        y_train = y[train_idx]
        C_val = C[val_idx]
        y_val = y[val_idx]
        
        # Precompute Gram matrix and cross-product for ridge solution
        CtC = C_train.T @ C_train
        Cty = C_train.T @ y_train
        
        fold_data.append((CtC, Cty, C_val, y_val))

    def solve_ridge(CtC, Cty, lam):
        """Solve ridge regression for Chebyshev coefficients."""
        if lam == 0.0:
            try:
                return np.linalg.solve(CtC, Cty)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(CtC, Cty, rcond=None)[0]
        else:
            A_mat = CtC + lam * np.eye(6)
            try:
                return np.linalg.solve(A_mat, Cty)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(A_mat, Cty, rcond=None)[0]

    def compute_cv_score(lam):
        """Compute mean CV MSE for a given lambda."""
        cv_errors = []
        for CtC, Cty, C_val, y_val in fold_data:
            c_cheb_train = solve_ridge(CtC, Cty, lam)
            pred_val = C_val @ c_cheb_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))
        if cv_errors:
            return np.mean(cv_errors)
        return np.inf

    # Lambda search
    # Chebyshev basis is better conditioned, so smaller lambdas might be optimal
    lambdas_coarse = np.concatenate([
        [0.0],
        np.logspace(-12, 4, 40)
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

    # Fine lambda search
    if best_lambda > 0:
        lower = max(best_lambda * 0.5, 1e-15)
        upper = best_lambda * 2.0
        lambdas_fine = np.logspace(np.log10(lower), np.log10(upper), 20)
    else:
        lambdas_fine = np.logspace(-12, -4, 20)

    for lam in lambdas_fine:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Final fit on all data
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        # Quick fit
        CtC_full = C.T @ C
        Cty_full = C.T @ y
        c_cheb_final = solve_ridge(CtC_full, Cty_full, best_lambda)
    else:
        CtC_full = C.T @ C
        Cty_full = C.T @ y
        c_cheb_final = solve_ridge(CtC_full, Cty_full, best_lambda)

    # Convert Chebyshev coefficients to scaled monomial coefficients
    # c_mono_scaled = M_cheb_to_mono @ c_cheb_final
    # From analysis:
    # c0 = c_cheb[0] - c_cheb[2] + c_cheb[4]
    # c1 = c_cheb[1] - 3*c_cheb[3] + 5*c_cheb[5]
    # c2 = 2*c_cheb[2] - 8*c_cheb[4]
    # c3 = 4*c_cheb[3] - 20*c_cheb[5]
    # c4 = 8*c_cheb[4]
    # c5 = 16*c_cheb[5]
    
    c_cheb = c_cheb_final
    c_mono_scaled = np.zeros(6)
    c_mono_scaled[0] = c_cheb[0] - c_cheb[2] + c_cheb[4]
    c_mono_scaled[1] = c_cheb[1] - 3.0*c_cheb[3] + 5.0*c_cheb[5]
    c_mono_scaled[2] = 2.0*c_cheb[2] - 8.0*c_cheb[4]
    c_mono_scaled[3] = 4.0*c_cheb[3] - 20.0*c_cheb[5]
    c_mono_scaled[4] = 8.0*c_cheb[4]
    c_mono_scaled[5] = 16.0*c_cheb[5]

    # Transform scaled monomial coefficients to original domain monomial coefficients
    coeff = T_trans @ c_mono_scaled

    if not np.all(np.isfinite(coeff)):
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

    return coeff
# EVOLVE-BLOCK-END
