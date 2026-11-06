"""Improved program: robust degree-five polynomial closure with regularization and degree selection."""
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

    # Handle edge cases
    if len(x) == 0 or len(y) == 0:
        return np.zeros(6)
    if len(x) != len(y):
        # Mismatch: take min length
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    if len(x) < 6:
        # Not enough points for degree 5; use lower degree
        deg = max(0, len(x) - 1)
        coeff = np.zeros(6)
        try:
            c = np.polynomial.polynomial.polyfit(x, y, deg)
            coeff[:deg+1] = c
        except Exception:
            pass
        return coeff

    # Normalize x to [-1, 1] for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        # All x values are the same
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)
    
    x_norm = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)

    # Try different degrees and regularization strengths
    best_coeff_norm = None
    best_score = np.inf

    # We'll evaluate on a held-out subset if data is large enough
    n = len(x)
    use_holdout = n > 100
    
    if use_holdout:
        # Split into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        n_val = max(10, n // 10)
        val_idx = indices[:n_val]
        tr_idx = indices[n_val:]
        x_tr, y_tr = x_norm[tr_idx], y[tr_idx]
        x_val, y_val = x_norm[val_idx], y[val_idx]
    else:
        x_tr, y_tr = x_norm, y
        x_val, y_val = x_norm, y  # Use all for validation (less reliable but no other choice)

    # Try degrees 0 through 5 with various regularization
    for deg in range(0, 6):
        # Try a few regularization strengths
        for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            if time.time() > deadline:
                break
            
            try:
                # Build Vandermonde matrix
                V = np.polynomial.polynomial.polyvander(x_tr, deg)
                
                if lam == 0.0:
                    # Unregularized least squares
                    if deg < len(x_tr):
                        coeff_norm, residuals, rank, sv = np.linalg.lstsq(V, y_tr, rcond=None)
                    else:
                        coeff_norm = np.zeros(deg + 1)
                        coeff_norm[0] = y_tr.mean()
                else:
                    # Regularized least squares (ridge)
                    # Minimize ||V*c - y||^2 + lam * ||c||^2
                    A = V.T @ V + lam * np.eye(deg + 1)
                    b = V.T @ y_tr
                    try:
                        coeff_norm = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        coeff_norm = np.linalg.lstsq(A, b, rcond=None)[0]
                
                # Check for finite values
                if not np.all(np.isfinite(coeff_norm)):
                    continue
                
                # Evaluate on validation set
                pred_val = np.polynomial.polynomial.polyval(x_val, coeff_norm)
                mse_val = np.mean((pred_val - y_val) ** 2)
                
                if not np.isfinite(mse_val):
                    continue
                
                if mse_val < best_score:
                    best_score = mse_val
                    best_coeff_norm = coeff_norm.copy()
            except Exception:
                continue
        
        if time.time() > deadline:
            break

    # If no valid coefficient was found, fall back to linear fit
    if best_coeff_norm is None:
        try:
            deg = min(1, len(x) - 1)
            c = np.polynomial.polynomial.polyfit(x_norm, y, deg)
            best_coeff_norm = np.zeros(6)
            best_coeff_norm[:deg+1] = c
        except Exception:
            best_coeff_norm = np.zeros(6)
            if len(y) > 0:
                best_coeff_norm[0] = y.mean()

    # Convert from normalized x to original x
    # x_norm = (x - mid) / half_range
    # So x = x_norm * half_range + mid
    # If p(x_norm) = sum c_i * x_norm^i, then p(x) = sum c_i * ((x - mid)/half_range)^i
    # = sum c_i * (1/half_range^i) * (x - mid)^i
    # Expand (x - mid)^i in terms of x to get coefficients in ascending powers of x
    
    mid = (x_min + x_max) / 2.0
    half_range = x_range / 2.0
    
    # Coefficients in normalized space: best_coeff_norm[i] for x_norm^i
    # We need to express as polynomial in x: sum_{k=0}^{5} d_k * x^k
    
    # Use polynomial composition: p(x) = q((x - mid)/half_range) where q is the normalized poly
    # q(t) = sum_{i=0}^{deg} c_i * t^i
    # t = (x - mid)/half_range = x/half_range - mid/half_range
    
    # Let a = 1/half_range, b = -mid/half_range
    # t = a*x + b
    # q(t) = sum_{i=0}^{deg} c_i * (a*x + b)^i
    # = sum_{i=0}^{deg} c_i * sum_{k=0}^{i} binom(i,k) * (a*x)^k * b^{i-k}
    # = sum_{k=0}^{deg} [sum_{i=k}^{deg} c_i * binom(i,k) * a^k * b^{i-k}] * x^k
    
    d = np.zeros(6)
    deg_actual = min(5, len(best_coeff_norm) - 1)
    a = 1.0 / half_range
    b = -mid / half_range
    
    for k in range(0, deg_actual + 1):
        s = 0.0
        for i in range(k, deg_actual + 1):
            if i >= len(best_coeff_norm):
                break
            ci = best_coeff_norm[i]
            if ci == 0.0:
                continue
            # binom(i, k)
            from math import comb
            binom_ik = comb(i, k)
            s += ci * binom_ik * (a ** k) * (b ** (i - k))
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
