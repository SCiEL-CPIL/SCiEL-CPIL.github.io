import numpy as np
import time
from math import comb

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0

    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Handle edge cases
    if len(x) == 0 or len(y) == 0:
        return np.zeros(6)
    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    
    n = len(x)
    if n < 6:
        # Fallback for very small data
        deg = max(0, n - 1)
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
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)
    
    mid = (x_min + x_max) / 2.0
    half_range = x_range / 2.0
    x_norm = (x - mid) / half_range

    # Prepare data for fitting
    # We want to fit in normalized space, then convert back.
    
    # Model 1: Odd Polynomial (x, x^3, x^5)
    # Model 2: Full Polynomial (1, x, x^2, x^3, x^4, x^5)
    
    # Split for validation if possible
    use_holdout = n > 50
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
    best_score = np.inf
    best_model_type = 'full' # Track which model was best

    # --- Fit Odd Model ---
    # Basis: [x, x^3, x^5]
    V_odd = np.column_stack([x_tr**1, x_tr**3, x_tr**5])
    
    # Try a few regularization strengths for the odd model
    for lam in [0.0, 1e-8, 1e-6, 1e-4]:
        if time.time() > deadline: break
        try:
            if lam == 0.0:
                if V_odd.shape[0] > V_odd.shape[1]:
                    c_odd, _, _, _ = np.linalg.lstsq(V_odd, y_tr, rcond=None)
                else:
                    c_odd = np.zeros(3)
                    c_odd[0] = np.dot(y_tr, x_tr) / np.dot(x_tr, x_tr) # Fallback
            else:
                A = V_odd.T @ V_odd + lam * np.eye(3)
                b = V_odd.T @ y_tr
                c_odd = np.linalg.solve(A, b)
            
            if not np.all(np.isfinite(c_odd)): continue
            
            # Evaluate on validation
            pred_val = c_odd[0]*x_val**1 + c_odd[1]*x_val**3 + c_odd[2]*x_val**5
            mse = np.mean((pred_val - y_val)**2)
            
            if mse < best_score:
                best_score = mse
                # Store coefficients in full 6-vector format (zeros for even powers)
                temp_coeff = np.zeros(6)
                temp_coeff[1] = c_odd[0]
                temp_coeff[3] = c_odd[1]
                temp_coeff[5] = c_odd[2]
                best_coeff_norm = temp_coeff
                best_model_type = 'odd'
        except Exception:
            continue

    # --- Fit Full Model ---
    V_full = np.polynomial.polynomial.polyvander(x_tr, 5)
    
    for lam in [0.0, 1e-8, 1e-6, 1e-4]:
        if time.time() > deadline: break
        try:
            if lam == 0.0:
                if V_full.shape[0] > V_full.shape[1]:
                    c_full, _, _, _ = np.linalg.lstsq(V_full, y_tr, rcond=None)
                else:
                    c_full = np.zeros(6)
                    c_full[0] = y_tr.mean()
            else:
                A = V_full.T @ V_full + lam * np.eye(6)
                b = V_full.T @ y_tr
                c_full = np.linalg.solve(A, b)
            
            if not np.all(np.isfinite(c_full)): continue
            
            # Evaluate on validation
            pred_val = np.polynomial.polynomial.polyval(x_val, c_full)
            mse = np.mean((pred_val - y_val)**2)
            
            # Heuristic: Prefer Odd model if it's close (within 10%) or better.
            # The "Knowledge" suggests Odd generalizes better to unseen regimes.
            # So we only switch to Full if it's significantly better on validation.
            threshold = best_score * 1.1 if best_model_type == 'odd' else np.inf
            
            if best_model_type == 'odd':
                if mse < best_score: # Only if Full is strictly better, but we bias against it
                    # Actually, let's stick to the knowledge: Odd is better.
                    # Only accept Full if it's dramatically better on train/val split
                    # to avoid overfitting Full.
                    if mse < best_score * 0.8: 
                        best_score = mse
                        best_coeff_norm = c_full
                        best_model_type = 'full'
            else:
                if mse < best_score:
                    best_score = mse
                    best_coeff_norm = c_full
                    best_model_type = 'full'
        except Exception:
            continue

    # If no valid coefficient was found, fall back to linear fit
    if best_coeff_norm is None:
        try:
            c = np.polynomial.polynomial.polyfit(x_norm, y, 1)
            best_coeff_norm = np.zeros(6)
            best_coeff_norm[:2] = c
        except Exception:
            best_coeff_norm = np.zeros(6)
            best_coeff_norm[0] = y.mean()

    # Convert from normalized x to original x
    # p(x_norm) = sum c_i x_norm^i
    # x_norm = (x - mid) / half_range
    # We need d_k such that sum d_k x^k = p((x-mid)/half_range)
    
    d = np.zeros(6)
    a = 1.0 / half_range
    b_const = -mid / half_range
    
    for k in range(0, 6):
        s = 0.0
        for i in range(k, 6):
            ci = best_coeff_norm[i]
            if ci == 0.0: continue
            # Coefficient of x^k in (a*x + b)^i is a^k * b^(i-k) * comb(i, k)
            s += ci * (a ** k) * (b_const ** (i - k)) * comb(i, k)
        d[k] = s

    # Ensure finite
    if not np.all(np.isfinite(d)):
        d = np.zeros(6)
        d[0] = y.mean()

    return d
