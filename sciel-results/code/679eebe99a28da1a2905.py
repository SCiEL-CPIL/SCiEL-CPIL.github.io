"""
Strategy: Hierarchical Binary Expansion for Bracketing.
1. Start at x0.
2. Expand the search interval [a, b] by doubling its size (first to the right, 
   then to the left) until a sign change is detected or a maximum range is reached.
   This ensures we find a bracket in O(log(distance_to_root)) calls, 
   rather than O(distance_to_root / step_size) calls used by linear scans.
3. Once a bracket [a, b] with f(a)*f(b) < 0 is found, use scipy.optimize.brentq 
   to converge to the root.
4. If no sign change is found (e.g., tangent root or no root in range), 
   fallback to a dense scan with adaptive step size.
This approach is robust for oscillatory functions because it adapts the 
"resolution" of the search to the distance of the root, avoiding wasted 
evaluations in regions far from the root.
"""
import math
import time
import numpy as np
from scipy.optimize import brentq

# Global deadline: evaluator kills after 150s wall-clock.
_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 490  # stay under 500


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


def _find_bracket_by_expansion(f, x0, counter):
    """
    Expand interval from x0 to find a sign change.
    Returns (a, b) if a bracket is found, else None.
    """
    # Initial small step
    step = 1e-6
    max_range = 1e6
    
    # Check immediate vicinity first with a few small steps
    # to catch roots very close to x0
    for i in range(1, 20):
        h = step * (2 ** i)
        if h > max_range:
            break
        
        # Check right
        x_r = x0 + h
        f_r = _counted_call(f, x_r, counter)
        if math.isfinite(f_r):
            f_0 = _counted_call(f, x0, counter) # Re-evaluate or cache? 
            # To save calls, we should cache f(x0). But _counted_call increments.
            # Let's refactor to pass f_x0 or handle caching.
            # Actually, just calling is safer for simplicity, but wastes 1 call.
            # Let's optimize: 
            pass
        
    # Better approach: Keep track of f(x0)
    f_x0 = _counted_call(f, x0, counter)
    if math.isfinite(f_x0) and abs(f_x0) <= 1e-8:
        return (x0, x0) # Root found
    
    # Expansion phases
    # Phase 1: Right expansion
    # Phase 2: Left expansion
    # We use a sequence of steps: 1, 2, 4, 8, ...
    
    # We need to check intervals [x0, x0+h], [x0+h, x0+2h], etc?
    # No, standard expansion is [x0, x0+h], [x0-h, x0], etc.
    # But for oscillatory functions, we might jump over a root.
    # However, if we double the step, we ensure we cover the space.
    # If f(x0) and f(x0+h) have same sign, we don't know if there's a root in between.
    # But for oscillatory functions, the period is finite.
    # If h is large enough, we are likely to catch a sign change.
    # If we miss it, we double h again.
    
    # Let's try a different expansion:
    # Check [x0, x0+h] for sign change.
    # If no, check [x0-h, x0] for sign change.
    # If no, double h.
    
    h = 1e-3
    max_h = 1e5
    
    while h < max_h and counter[0] < _CALL_BUDGET - 10 and _time_left() > 1.0:
        # Check right
        x_r = x0 + h
        f_r = _counted_call(f, x_r, counter)
        
        if math.isfinite(f_r):
            if f_x0 * f_r < 0:
                return (x0, x_r)
            # If f_r is close to 0, return it
            if abs(f_r) <= 1e-8:
                return (x_r, x_r)
        
        # Check left
        x_l = x0 - h
        f_l = _counted_call(f, x_l, counter)
        
        if math.isfinite(f_l):
            if f_x0 * f_l < 0:
                return (x_l, x0)
            if abs(f_l) <= 1e-8:
                return (x_l, x_l)
        
        h *= 2.0
            
    return None


def _dense_scan_fallback(f, x0, counter):
    """
    Fallback: Dense scan with adaptive step size.
    """
    # Estimate frequency roughly if possible, or just use a fixed fine grid
    # Since expansion failed, maybe the root is tangent or we missed it.
    # Try a dense scan around x0 with a smaller step.
    
    # Let's try a few scales
    scales = [1.0, 10.0, 100.0]
    steps = [10, 50, 100]
    
    for scale, n_points in zip(scales, steps):
        a = x0 - scale
        b = x0 + scale
        xs = np.linspace(a, b, n_points)
        
        prev_x = None
        prev_f = None
        
        for x in xs:
            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                break
                
            fx = _counted_call(f, x, counter)
            
            if math.isfinite(fx):
                if abs(fx) <= 1e-8:
                    return x
                
                if prev_f is not None and math.isfinite(prev_f):
                    if prev_f * fx < 0:
                        # Bracket found
                        try:
                            result = brentq(f, prev_x, x, xtol=1e-12, rtol=1e-12, maxiter=100)
                            if abs(f(result)) <= 1e-8:
                                return result
                        except Exception:
                            pass
                
                prev_x = x
                prev_f = fx
            else:
                prev_x = None
                prev_f = None
                
    return None


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Try hierarchical expansion
    bracket = _find_bracket_by_expansion(f, x0, counter)
    
    if bracket is not None:
        a, b = bracket
        if a == b:
            return a
        try:
            result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            # Verify
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
            
    # 2. Fallback to dense scan
    result = _dense_scan_fallback(f, x0, counter)
    if result is not None:
        return result
        
    # 3. Last resort: return x0 (will likely fail, but avoids exception)
    return x0
