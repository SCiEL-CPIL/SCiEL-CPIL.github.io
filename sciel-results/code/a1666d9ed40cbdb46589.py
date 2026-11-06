"""Combined program: robust degree-five polynomial closure with normalization, regularization, and direct-space evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove any NaN or Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Handle edge cases
    n = len(x)
    if n == 0:
        return np.zeros(6)
    if n < 6:
        deg = max(0, n - 1)
        try:
            c = np.polynomial.polynomial.polyfit(x, y, deg)
            coeff = np.zeros(6)
            coeff[:deg+1] = c
            return coeff
        except Exception:
            return np.array([y.mean() if n > 0 else 0.0] + [0.0]*5)

    # Check if x_range is degenerate
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([y.mean()] + [0.0]*5)

    # Normalize x to [-1, 1] for better conditioning during fitting
    mid = (x_min + x_max) / 2.0
    half_range = x_range / 2.0
    x_norm = (x - mid) / half_range

    # Split into train/val for model selection if enough data
    use_holdout = n > 100
    if use_holdout:
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        n_val = max(10, n // 10)
        val_idx = indices[:n_val]
        tr_idx = indices[n_val:]
        x_tr, y_tr = x_norm[tr_idx], y[tr_idx]
        x_val, y_val = x_norm[val_idx], y[val_idx]
    else:
        x_tr, y_tr = x_norm, y
        x_val, y_val = x_norm, y

    best_coeff_norm = None
    best_val_mse = np.inf

    # Try degrees 0 through 5
    for deg in range(0, 6):
        if time.time() > deadline:
            break

        try:
            # Build Vandermonde matrix in normalized space
            V = np.polynomial.polynomial.polyvander(x_tr, deg)
            
            # Compute normal equations
            A = V.T @ V
            b = V.T @ y_tr
            
            # Try unregularized first
            try:
                coeff_norm, residuals, rank, sv = np.linalg.lstsq(V, y_tr, rcond=None)
                if np.all(np.isfinite(coeff_norm)):
                    pred_val = np.polynomial.polynomial.polyval(x_val, coeff_norm)
                    mse_val = np.mean((pred_val - y_val) ** 2)
                    if np.isfinite(mse_val) and mse_val < best_val_mse:
                        best_val_mse = mse_val
                        best_coeff_norm = coeff_norm.copy()
            except Exception:
                pass
            
            # Try small ridge regularization
            for lam in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]:
                if time.time() > deadline:
                    break
                try:
                    A_reg = A + lam * np.eye(deg + 1)
                    coeff_reg = np.linalg.solve(A_reg, b)
                    if not np.all(np.isfinite(coeff_reg)):
                        continue
                    pred_val = np.polynomial.polynomial.polyval(x_val, coeff_reg)
                    mse_val = np.mean((pred_val - y_val) ** 2)
                    if np.isfinite(mse_val) and mse_val < best_val_mse:
                        best_val_mse = mse_val
                        best_coeff_norm = coeff_reg.copy()
                except Exception:
                    continue

        except Exception:
            continue

    # If no valid coefficient was found, fall back to linear fit in normalized space
    if best_coeff_norm is None:
        try:
            deg = min(1, len(x) - 1)
            c = np.polynomial.polynomial.polyfit(x_norm, y, deg)
            best_coeff_norm = np.zeros(deg + 1)
            best_coeff_norm[:deg+1] = c
        except Exception:
            best_coeff_norm = np.array([y.mean()])

    # Convert from normalized x to original x
    # x_norm = (x - mid) / half_range
    # So x = x_norm * half_range + mid
    # If p(x_norm) = sum c_i * x_norm^i, then p(x) = sum c_i * ((x - mid)/half_range)^i
    # = sum c_i * (1/half_range^i) * (x - mid)^i
    # Expand (x - mid)^i in terms of x to get coefficients in ascending powers of x
    
    d = np.zeros(6)
    deg_actual = min(5, len(best_coeff_norm) - 1)
    a = 1.0 / half_range
    b_const = -mid / half_range
    
    for k in range(0, deg_actual + 1):
        s = 0.0
        for i in range(k, deg_actual + 1):
            if i >= len(best_coeff_norm):
                break
            ci = best_coeff_norm[i]
            if ci == 0.0:
                continue
            binom_ik = comb(i, k)
            s += ci * binom_ik * (a ** k) * (b_const ** (i - k))
        d[k] = s

    # Ensure all values are finite
    if not np.all(np.isfinite(d)):
        # Fallback: simple linear fit in original space
        try:
            deg = min(1, len(x) - 1)
            c = np.polynomial.polynomial.polyfit(x, y, deg)
            d = np.zeros(6)
            d[:deg+1] = c
        except Exception:
            d = np.zeros(6)
            if len(y) > 0:
                d[0] = y.mean()

    return d
# EVOLVE-BLOCK-END
