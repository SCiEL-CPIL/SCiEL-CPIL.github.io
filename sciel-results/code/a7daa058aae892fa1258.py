"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _nCr(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0:
        return 1
    r = min(r, n - r)
    res = 1
    for i in range(r):
        res = res * (n - i) // (i + 1)
    return res

def _convert_coeffs_to_original(a, mu, sigma, deg=5):
    """
    Convert polynomial coefficients 'a' for the variable z = (x-mu)/sigma
    to coefficients 'b' for the original variable x.
    y = sum(a_j * z^j) = sum(b_i * x^i)
    """
    b = np.zeros(deg + 1)
    for j in range(deg + 1):
        if a[j] == 0:
            continue
        coeff_a = a[j] / (sigma ** j)
        for k in range(j + 1):
            binom_val = _nCr(j, k)
            term_coeff = coeff_a * binom_val * ((-mu) ** (j - k))
            b[k] += term_coeff
    return b

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]

    if len(x) < 10:
        return np.array([np.mean(y)] + [0.0]*5)

    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-8:
        sigma = 1.0
    z = (x - mu) / sigma

    deg = 5
    n = len(z)
    V_full = np.polynomial.polynomial.polyvander(z, deg)

    # Three-stage lambda search: coarse, fine, very-fine
    lambdas_coarse = np.logspace(-6, 6, 60)

    n_folds = 5
    if n < n_folds * 2:
        n_folds = max(2, n // 2)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        folds.append((indices[start:end], np.concatenate([indices[:start], indices[end:]])))

    best_cv_mse = np.inf
    best_lambda = 1.0

    # Precompute V^T V and V^T y for each fold
    fold_data = []
    for val_idx, train_idx in folds:
        if len(train_idx) < deg + 1:
            continue
        V_tr = V_full[train_idx]
        y_tr = y[train_idx]
        V_val = V_full[val_idx]
        y_val = y[val_idx]
        VtV = V_tr.T @ V_tr
        Vty = V_tr.T @ y_tr
        fold_data.append((VtV, Vty, V_val, y_val))

    I = np.eye(deg + 1)

    # Stage 1: Coarse search
    for lam in lambdas_coarse:
        if time.time() > deadline - 30.0:
            break
        cv_mse_sum = 0.0
        valid_folds = 0
        for VtV, Vty, V_val, y_val in fold_data:
            try:
                A = VtV + lam * I
                c = np.linalg.solve(A, Vty)
                if not np.all(np.isfinite(c)):
                    continue
                pred = V_val @ c
                mse = np.mean((pred - y_val) ** 2)
                cv_mse_sum += mse
                valid_folds += 1
            except Exception:
                continue
        if valid_folds > 0:
            avg_mse = cv_mse_sum / valid_folds
            if avg_mse < best_cv_mse:
                best_cv_mse = avg_mse
                best_lambda = lam

    # Stage 2: Fine search around best_lambda
    if time.time() < deadline - 25.0:
        ratio = 1.5
        lambdas_fine = np.logspace(np.log10(best_lambda / ratio), np.log10(best_lambda * ratio), 40)
        lambdas_fine = np.unique(lambdas_fine)

        for lam in lambdas_fine:
            if time.time() > deadline - 15.0:
                break
            cv_mse_sum = 0.0
            valid_folds = 0
            for VtV, Vty, V_val, y_val in fold_data:
                try:
                    A = VtV + lam * I
                    c = np.linalg.solve(A, Vty)
                    if not np.all(np.isfinite(c)):
                        continue
                    pred = V_val @ c
                    mse = np.mean((pred - y_val) ** 2)
                    cv_mse_sum += mse
                    valid_folds += 1
                except Exception:
                    continue
            if valid_folds > 0:
                avg_mse = cv_mse_sum / valid_folds
                if avg_mse < best_cv_mse:
                    best_cv_mse = avg_mse
                    best_lambda = lam

    # Stage 3: Very fine search around best_lambda
    if time.time() < deadline - 10.0:
        ratio = 1.15
        lambdas_vfine = np.logspace(np.log10(best_lambda / ratio), np.log10(best_lambda * ratio), 30)
        lambdas_vfine = np.unique(lambdas_vfine)

        for lam in lambdas_vfine:
            if time.time() > deadline - 5.0:
                break
            cv_mse_sum = 0.0
            valid_folds = 0
            for VtV, Vty, V_val, y_val in fold_data:
                try:
                    A = VtV + lam * I
                    c = np.linalg.solve(A, Vty)
                    if not np.all(np.isfinite(c)):
                        continue
                    pred = V_val @ c
                    mse = np.mean((pred - y_val) ** 2)
                    cv_mse_sum += mse
                    valid_folds += 1
                except Exception:
                    continue
            if valid_folds > 0:
                avg_mse = cv_mse_sum / valid_folds
                if avg_mse < best_cv_mse:
                    best_cv_mse = avg_mse
                    best_lambda = lam

    # Refit on all data with best_lambda
    # Use lstsq when lambda is very small to avoid ill-conditioning
    if best_lambda < 1e-10:
        try:
            c_std, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
            if np.all(np.isfinite(c_std)):
                b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
                if np.all(np.isfinite(b_orig)):
                    return b_orig
        except Exception:
            pass
    else:
        A = V_full.T @ V_full + best_lambda * I
        b_vec = V_full.T @ y
        try:
            c_std = np.linalg.solve(A, b_vec)
            if np.all(np.isfinite(c_std)):
                b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
                if np.all(np.isfinite(b_orig)):
                    return b_orig
        except Exception:
            pass

    # Fallback: Unregularized fit
    try:
        c_std, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
        if np.all(np.isfinite(c_std)):
            b_orig = _convert_coeffs_to_original(c_std, mu, sigma, deg)
            if np.all(np.isfinite(b_orig)):
                return b_orig
    except Exception:
        pass

    return np.array([np.mean(y)] + [0.0]*5)

# EVOLVE-BLOCK-END
