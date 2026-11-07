"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    """
    Find a root of f starting near x0 by minimizing f(x)^2 over dynamically
    expanded bounded intervals using SciPy's minimize_scalar.
    
    Strategy:
    1. Adaptive coarse scan to locate sign changes and small |f(x)|
    2. Use golden-section / bounded minimization on best bracket
    3. Refine with tighter tolerances
    """
    x0 = float(x0)
    call_count = 0
    call_limit = 470  # Stay well under 500
    
    def counted_f(x):
        nonlocal call_count
        if call_count >= call_limit:
            raise RuntimeError("Call limit exceeded")
        call_count += 1
        val = f(x)
        if not np.isfinite(val):
            return float('inf')
        return val
    
    # Step 1: Adaptive scan to find sign changes and small values
    best_x = x0
    best_abs = abs(counted_f(x0))
    
    if best_abs <= 1e-8:
        return best_x
    
    # Track sign changes for bisection fallback
    sign_change_pairs = []
    prev_x = x0
    prev_fx = best_abs if best_abs > 0 else 1e-10  # use signed value
    # Actually we need signed f for sign changes
    prev_fx_signed = counted_f(x0) if call_count < call_limit else float('inf')
    
    # Multi-scale scan
    scan_configs = [
        # (radius, num_points)
        (0.01, 5), (0.05, 5), (0.1, 7), (0.5, 7), (1.0, 9),
        (2.0, 9), (5.0, 11), (10.0, 11), (50.0, 13), (100.0, 13),
        (500.0, 15), (1000.0, 15),
    ]
    
    for radius, npts in scan_configs:
        if call_count >= call_limit - 60:
            break
        for i in range(npts):
            if call_count >= call_limit - 60:
                break
            t = i / (npts - 1) if npts > 1 else 0.5
            x_sample = x0 - radius + 2 * radius * t
            try:
                fx = counted_f(x_sample)
            except RuntimeError:
                break
            abs_fx = abs(fx)
            if abs_fx < best_abs:
                best_abs = abs_fx
                best_x = x_sample
                if best_abs <= 1e-8:
                    return best_x
            # Check for sign change with previous point
            if np.isfinite(prev_fx_signed) and np.isfinite(fx):
                if prev_fx_signed * fx < 0:
                    sign_change_pairs.append((prev_x, x_sample))
            prev_x = x_sample
            prev_fx_signed = fx
        if best_abs <= 1e-8:
            return best_x
    
    # Step 2: Try bisection on any sign change pairs found
    for (a, b) in sign_change_pairs:
        if call_count >= call_limit - 20:
            break
        fa = counted_f(a)
        fb = counted_f(b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa * fb > 0:
            continue
        # Bisection
        for _ in range(50):
            if call_count >= call_limit - 5:
                break
            mid = (a + b) / 2.0
            fm = counted_f(mid)
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        # Check final
        if call_count < call_limit:
            fm = counted_f((a + b) / 2.0)
            if abs(fm) <= 1e-8:
                return (a + b) / 2.0
    
    # Step 3: Use minimize_scalar with expanding brackets around best_x
    base_width = max(1e-3, min(1.0, best_abs ** 0.25))
    if not np.isfinite(base_width) or base_width <= 0:
        base_width = 1e-2
    
    widths = [base_width, base_width * 2, base_width * 4, base_width * 8,
              base_width * 16, base_width * 32, base_width * 64]
    
    for w in widths:
        if call_count >= call_limit - 25:
            break
        a = best_x - w
        b = best_x + w
        try:
            res = minimize_scalar(
                lambda x: counted_f(x) ** 2,
                bounds=(a, b),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 40}
            )
            x_candidate = res.x
            if np.isfinite(x_candidate):
                fx_candidate = abs(counted_f(x_candidate))
                if fx_candidate < best_abs:
                    best_abs = fx_candidate
                    best_x = x_candidate
                    if best_abs <= 1e-8:
                        return best_x
        except (RuntimeError, Exception):
            continue
    
    # Step 4: Broader search from x0
    if call_count < call_limit - 15:
        for w in [5.0, 20.0, 100.0, 500.0]:
            if call_count >= call_limit - 10:
                break
            a = x0 - w
            b = x0 + w
            try:
                res = minimize_scalar(
                    lambda x: counted_f(x) ** 2,
                    bounds=(a, b),
                    method='bounded',
                    options={'xatol': 1e-8, 'maxiter': 25}
                )
                x_candidate = res.x
                if np.isfinite(x_candidate):
                    fx_candidate = abs(counted_f(x_candidate))
                    if fx_candidate < best_abs:
                        best_abs = fx_candidate
                        best_x = x_candidate
                        if best_abs <= 1e-8:
                            return best_x
            except (RuntimeError, Exception):
                continue
    
    return best_x
# EVOLVE-BLOCK-END
