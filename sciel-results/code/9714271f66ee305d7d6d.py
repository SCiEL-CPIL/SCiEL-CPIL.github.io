"""Combined program: robust degree-five polynomial closure with direct-space fitting and adaptive regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


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

    # Split into train/val for model selection if enough data
    use_holdout = n > 100
    if use_holdout:
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        n_val = max(10, n // 10)
        val_idx = indices[:n_val]
        tr_idx = indices[n_val:]
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_val, y_val = x[val_idx], y[val_idx]
    else:
        x_tr, y_tr = x, y
        x_val, y_val = x, y

    best_coeff = None
    best_val_mse = np.inf

    # Try degrees 0 through 5
    for deg in range(0, 6):
        if time.time() > deadline:
            break

        try:
            # Build Vandermonde matrix in original x-space (ascending powers)
            V = np.polynomial.polynomial.polyvander(x_tr, deg)
            
            # Compute normal equations
            A = V.T @ V
            b = V.T @ y_tr
            
            # Check condition number
            try:
                cond_A = np.linalg.cond(A)
            except Exception:
                cond_A = np.inf
            
            # Try unregularized first
            if cond_A < 1e14:
                try:
                    coeff_norm, residuals, rank, sv = np.linalg.lstsq(V, y_tr, rcond=None)
                    if np.all(np.isfinite(coeff_norm)):
                        pred_val = np.polynomial.polynomial.polyval(x_val, coeff_norm)
                        mse_val = np.mean((pred_val - y_val) ** 2)
                        if np.isfinite(mse_val) and mse_val < best_val_mse:
                            best_val_mse = mse_val
                            best_coeff = np.zeros(6)
                            best_coeff[:deg+1] = coeff_norm
                except Exception:
                    pass
            
            # Try small ridge regularization if needed or as alternative
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
                        best_coeff = np.zeros(6)
                        best_coeff[:deg+1] = coeff_reg
                except Exception:
                    continue

        except Exception:
            continue

    # If no valid coefficient was found, fall back to simple fits
    if best_coeff is None:
        # Try linear fit
        try:
            c = np.polynomial.polynomial.polyfit(x, y, 1)
            best_coeff = np.zeros(6)
            best_coeff[:2] = c
        except Exception:
            # Constant fit
            best_coeff = np.zeros(6)
            best_coeff[0] = y.mean()

    # Final check: ensure all values are finite
    if not np.all(np.isfinite(best_coeff)):
        best_coeff = np.zeros(6)
        best_coeff[0] = y.mean() if n > 0 else 0.0

    return best_coeff
# EVOLVE-BLOCK-END
