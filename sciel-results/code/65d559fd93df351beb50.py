"""
Strategy: Multi-scale frequency-aware root finding.
1. Estimate the dominant frequency/period of the oscillatory function using
   a small number of coarse probes at increasing distances from x0.
2. Once a plausible period T is estimated, densely sample [x0 - 2T, x0 + 2T]
   to find sign changes (roots).
3. Refine each candidate bracket with a fast Newton/secant hybrid.
This differs from the current program by not relying primarily on local
exponential expansion from x0, which can miss roots if the initial step is
comparable to or larger than the wavelength. Instead, it globally adapts
the search scale to the function's oscillatory nature.
"""
import math
import time

_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490


def _time_left():
    return _DEADLINE - time.time()


def _counted_call(f, x, counter):
    """Call f(x), track count, guard against exceptions / non-finite."""
    counter[0] += 1
    try:
        val = f(x)
    except Exception:
        return float('nan')
    if not math.isfinite(val):
        return val
    return val


def _estimate_period(f, x0, counter, max_probes=12):
    """
    Estimate the dominant period by looking for sign changes at increasing
    distances. Returns (period, direction) or None.
    """
    # Probe at distances: 0.1, 0.2, 0.4, 0.8, 1.6, ...
    # For each distance d, check f(x0) and f(x0 +/- d).
    # If we see a sign change, the period is roughly 2*d (if half-period) or d.
    # We'll collect all sign-change distances and pick the smallest consistent one.
    
    f0 = _counted_call(f, x0, counter)
    if not math.isfinite(f0):
        return None
    
    sign_changes = []
    d = 0.1
    for _ in range(max_probes):
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 20:
            break
            
        fp = _counted_call(f, x0 + d, counter)
        fm = _counted_call(f, x0 - d, counter)
        
        if math.isfinite(fp) and f0 * fp < 0:
            # Sign change between x0 and x0+d. Period is roughly 2*d? Or d?
            # If f(x0) and f(x0+d) have opposite signs, there's a root in between.
            # The distance to the next root of the same sign might be ~2d.
            sign_changes.append(d)
            
        if math.isfinite(fm) and f0 * fm < 0:
            sign_changes.append(d)
            
        d *= 2.0
        if d > 100.0:
            break
    
    if not sign_changes:
        return None
    
    # Use the smallest sign-change distance as a proxy for half-period
    # Period T ≈ 2 * min(sign_changes)
    T = 2.0 * min(sign_changes)
    return T


def _newton_secant(f, x0, counter, max_iter=50, tol=1e-12):
    """
    Fast Newton/secant hybrid for polishing a root.
    Returns x or None if it fails.
    """
    x = float(x0)
    fx = _counted_call(f, x, counter)
    if not math.isfinite(fx):
        return None
    if abs(fx) <= 1e-8:
        return x
    
    # Secant method requires two points
    h = 1e-6
    x1 = x + h
    f1 = _counted_call(f, x1, counter)
    if not math.isfinite(f1):
        return None
    
    for _ in range(max_iter):
        if _time_left() < 0.5 or counter[0] >= _CALL_BUDGET:
            break
            
        if abs(f1 - fx) < 1e-15:
            # Derivative ~ 0, use Newton with finite difference
            h_new = 1e-6
            f1 = _counted_call(f, x + h_new, counter)
            if not math.isfinite(f1):
                break
            dfdx = (f1 - fx) / h_new
        else:
            dfdx = (f1 - fx) / (x1 - x)
            
        if abs(dfdx) < 1e-15:
            break
            
        x_new = x - fx / dfdx
        
        if not math.isfinite(x_new):
            break
            
        # Damping if step is too large
        if abs(x_new - x) > 10.0 * abs(x) + 10.0:
            x_new = x + 0.5 * (x_new - x)
            
        fx_new = _counted_call(f, x_new, counter)
        if not math.isfinite(fx_new):
            break
            
        if abs(fx_new) <= 1e-8:
            return x_new
            
        x, fx = x_new, fx_new
        x1, f1 = x, fx
    
    if abs(fx) <= 1e-8:
        return x
    return None


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy: Estimate period, densely scan for sign changes, refine with Newton/secant.
    """
    counter = [0]
    x0 = float(x0)
    
    # First, check if x0 is already a root
    fx0 = _counted_call(f, x0, counter)
    if math.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0
    
    # Estimate period
    T = _estimate_period(f, x0, counter)
    
    if T is None:
        # Fallback: use a default period guess
        T = 1.0
    
    # Densely scan [x0 - 2T, x0 + 2T] for sign changes
    # Number of points: ~100 per period, so 200 points total
    n_points = 200
    a = x0 - 2.0 * T
    b = x0 + 2.0 * T
    h = (b - a) / n_points
    
    prev_x = a
    prev_fx = _counted_call(f, prev_x, counter)
    
    candidates = []
    
    for i in range(1, n_points + 1):
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
            
        x = a + i * h
        fx = _counted_call(f, x, counter)
        
        if math.isfinite(prev_fx) and math.isfinite(fx):
            if prev_fx * fx < 0:
                # Sign change: potential root in [prev_x, x]
                candidates.append((prev_x, x))
            elif abs(fx) <= 1e-8:
                # Direct hit
                return x
        
        prev_x = x
        prev_fx = fx
    
    # Refine each candidate bracket with Newton/secant
    # Start from the one closest to x0
    candidates.sort(key=lambda c: abs((c[0] + c[1]) / 2.0 - x0))
    
    for a_c, b_c in candidates:
        if _time_left() < 0.5 or counter[0] >= _CALL_BUDGET - 5:
            break
            
        # Use midpoint as initial guess for Newton/secant
        x_mid = (a_c + b_c) / 2.0
        result = _newton_secant(f, x_mid, counter)
        
        if result is not None:
            # Verify
            f_res = _counted_call(f, result, counter)
            if math.isfinite(f_res) and abs(f_res) <= 1e-8:
                return result
    
    # If no candidates found or all failed, try a few direct Newton attempts
    # from x0 with different initial steps
    for delta in [0.01, -0.01, 0.1, -0.1, 1.0, -1.0]:
        if _time_left() < 0.5 or counter[0] >= _CALL_BUDGET - 5:
            break
            
        x_try = x0 + delta
        result = _newton_secant(f, x_try, counter)
        
        if result is not None:
            f_res = _counted_call(f, result, counter)
            if math.isfinite(f_res) and abs(f_res) <= 1e-8:
                return result
    
    # Last resort: return x0
    return x0
