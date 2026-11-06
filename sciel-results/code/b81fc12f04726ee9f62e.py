"""Stronger degree-five polynomial closure with orthonormal-basis selection, ridge sweep, and cross-validated model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    t0 = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n == 0:
        return np.zeros(6)

    if n < 6:
        deg = max(0, min(5, n - 1))
        try:
            c = np.polynomial.polynomial.polyfit(x, y, deg)
            coeff = np.zeros(6)
            coeff[:deg + 1] = c
            if not np.all(np.isfinite(coeff)):
                coeff[0] = y.mean()
                coeff[1:] = 0.0
            return coeff
        except Exception:
            c = np.zeros(6)
            c[0] = y.mean()
            return c

    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min

    if x_range < 1e-12:
        c = np.zeros(6)
        c[0] = y.mean()
        return c

    # Center and scale x for stable polynomial basis
    x_c = x - x_min
    x_s = x_c / (x_range + 1e-300)

    # Precompute orthonormal-like basis matrix via QR on shifted Vandermonde
    # We'll work in the original monomial basis but solve in a stabilized way

    # K-fold cross-validation for model selection
    k = min(5, max(2, n // 1000))
    if k < 2:
        k = 2
    rng = np.random.default_rng(12345)
    fold_ids = rng.permutation(n) % k

    best_score = np.inf
    best_coeff = np.zeros(6)

    # Try degrees 0..5
    for deg in range(0, 6):
        if time.time() > deadline:
            break

        d = deg + 1
        try:
            # Build Vandermonde in scaled x
            V_full = np.polynomial.polynomial.polyvander(x_s, deg)

            # Cross-validated MSE
            fold_mses = np.zeros(k)
            for f in range(k):
                val_mask = (fold_ids == f)
                tr_mask = ~val_mask
                if np.sum(tr_mask) < d:
                    continue
                V_tr = V_full[tr_mask]
                y_tr = y[tr_mask]
                V_val = V_full[val_mask]
                y_val = y[val_mask]

                # Ridge solve
                best_lam_mse = np.inf
                best_lam_coeff = None
                for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]:
                    if time.time() > deadline:
                        break
                    try:
                        A = V_tr.T @ V_tr
                        b = V_tr.T @ y_tr
                        if lam == 0.0:
                            if np.linalg.cond(A) < 1e12:
                                coeff_l = np.linalg.solve(A, b)
                            else:
                                coeff_l, _, _, _ = np.linalg.lstsq(V_tr, y_tr, rcond=None)
                        else:
                            A_reg = A + lam * np.eye(d)
                            coeff_l = np.linalg.solve(A_reg, b)

                        if not np.all(np.isfinite(coeff_l)):
                            continue

                        pred = V_val @ coeff_l
                        mse = np.mean((pred - y_val) ** 2)
                        if np.isfinite(mse) and mse < best_lam_mse:
                            best_lam_mse = mse
                            best_lam_coeff = coeff_l
                    except Exception:
                        continue

                if best_lam_mse < np.inf:
                    fold_mses[f] = best_lam_mse

            cv_mse = np.mean(fold_mses)
            if not np.isfinite(cv_mse):
                continue

            if cv_mse < best_score:
                # Refit on all data with best lambda
                best_lam_mse = np.inf
                best_lam_coeff = None
                for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]:
                    try:
                        A = V_full.T @ V_full
                        b = V_full.T @ y
                        if lam == 0.0:
                            if np.linalg.cond(A) < 1e12:
                                coeff_l = np.linalg.solve(A, b)
                            else:
                                coeff_l, _, _, _ = np.linalg.lstsq(V_full, y, rcond=None)
                        else:
                            A_reg = A + lam * np.eye(d)
                            coeff_l = np.linalg.solve(A_reg, b)

                        if not np.all(np.isfinite(coeff_l)):
                            continue

                        pred = V_full @ coeff_l
                        mse = np.mean((pred - y) ** 2)
                        if np.isfinite(mse) and mse < best_lam_mse:
                            best_lam_mse = mse
                            best_lam_coeff = coeff_l
                    except Exception:
                        continue

                if best_lam_coeff is not None:
                    # Convert back to original x-space coefficients
                    # V_full is in powers of x_s = (x - x_min) / x_range
                    # We need coefficients c_i such that sum(c_i * x**i) approximates y
                    # Build transformation: express (x_s)^j in terms of x^i
                    # x_s = (x - x_min) / x_range
                    # So (x_s)^j = (1/x_range^j) * (x - x_min)^j
                    # Expand (x - x_min)^j = sum_{i=0}^{j} C(j,i) * x^i * (-x_min)^(j-i)
                    # So coeff_orig[i] = sum_{j=i}^{deg} coeff_l[j] * (1/x_range^j) * C(j,i) * (-x_min)^(j-i)

                    coeff_orig = np.zeros(6)
                    for j in range(d):
                        if coeff_l[j] == 0.0:
                            continue
                        scale = (1.0 / (x_range ** j)) if x_range != 0 else 0.0
                        for i in range(j + 1):
                            # C(j, i) * (-x_min)^(j-i)
                            binom = _binom(j, i)
                            term = binom * ((-x_min) ** (j - i))
                            coeff_orig[i] += coeff_l[j] * scale * term

                    if np.all(np.isfinite(coeff_orig)):
                        best_score = cv_mse
                        best_coeff = coeff_orig

        except Exception:
            continue

    # Final validation
    if not np.all(np.isfinite(best_coeff)):
        try:
            c = np.polynomial.polynomial.polyfit(x, y, min(5, n - 1))
            best_coeff = np.zeros(6)
            best_coeff[:min(5, n - 1) + 1] = c
        except Exception:
            best_coeff = np.zeros(6)
            best_coeff[0] = y.mean()

    return best_coeff


def _binom(n, k):
    """Compute binomial coefficient C(n, k) using integer arithmetic."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return result
# EVOLVE-BLOCK-END
