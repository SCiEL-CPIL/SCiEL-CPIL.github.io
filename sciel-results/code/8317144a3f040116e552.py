"""
Strategy: Multi-Scale Adaptive Wavelength Detection.
Unlike the current program which expands from x0 with a fixed geometric progression,
this approach first estimates the local oscillation frequency (wavelength) by probing
the function at a few distant points. It then uses this estimated wavelength to
perform a targeted, dense local search for a sign change, followed by Brent's method.
This is structurally different because it adapts the search scale to the function's
characteristic frequency rather than assuming a generic expansion factor.
"""
import math
import time
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

def _estimate_wavelength(f, x0, counter):
    """
    Estimate the local wavelength of the oscillatory function.
    We probe points at increasing distances from x0 to find the first sign change
    or significant variation, which gives us a scale for the oscillation period.
    Returns an estimate of half-period or step size to use for local search.
    """
    # Probe at various scales
    scales = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    f0 = _counted_call(f, x0, counter)
    if not math.isfinite(f0):
        return None
        
    for scale in scales:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
            
        x1 = x0 + scale
        x2 = x0 - scale
        
        f1 = _counted_call(f, x1, counter)
        f2 = _counted_call(f, x2, counter)
        
        # If we see a sign change at this scale, the wavelength is roughly 2*scale or less
        # If we see large variation without sign change, it's also informative.
        if math.isfinite(f1) and math.isfinite(f2):
            # Heuristic: if sign changes, wavelength is ~ 2*scale. 
            # If values are similar, wavelength is larger.
            # If values are very different, wavelength is smaller.
            
            # Check for sign change
            if f0 * f1 < 0 or f0 * f2 < 0:
                return scale
            elif f1 * f2 < 0:
                return scale / 2.0
            
            # Check magnitude of change
            delta1 = abs(f1 - f0)
            delta2 = abs(f2 - f0)
            
            # If the function changed significantly, the scale is too big
            # If it didn't change, the scale might be too small (or we are at a flat spot)
            # This is tricky without knowing the amplitude. 
            # Let's assume if |f(x+scale) - f(x)| is large, scale is comparable to wavelength.
            # But we don't know the amplitude.
            
            # Alternative: just return the scale where we first see a sign change.
            # If no sign change is found in the coarse scan, we might be in a region
            # with very high frequency or we need a finer scan.
            
    # If no sign change found in coarse scan, assume a default small step
    # This might be dangerous if the function is very flat, but oscillatory functions
    # usually have sign changes.
    return 0.1

def _local_search(f, x0, step, counter):
    """
    Perform a dense local search around x0 using the estimated step size.
    We scan a window of +/- 5*step with a resolution of step/10.
    """
    # If step is None or non-finite, use default
    if step is None or not math.isfinite(step) or step <= 0:
        step = 0.1
        
    # Ensure step is not too small or too large
    step = max(1e-4, min(step, 100.0))
    
    # Scan window: [x0 - 5*step, x0 + 5*step]
    # Resolution: step / 10
    res = step / 10.0
    
    start = x0 - 5 * step
    end = x0 + 5 * step
    
    n_steps = int((end - start) / res) + 1
    
    prev_x = None
    prev_f = None
    
    for i in range(n_steps):
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
            break
            
        x = start + i * res
        fx = _counted_call(f, x, counter)
        
        if not math.isfinite(fx):
            prev_x = x
            prev_f = fx
            continue
            
        if prev_f is not None and math.isfinite(prev_f):
            if prev_f * fx < 0:
                # Found a bracket [prev_x, x]
                return (prev_x, x)
                
        prev_x = x
        prev_f = fx
        
    return None

def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy: 
    1. Estimate wavelength by coarse probing.
    2. Perform dense local search using estimated wavelength.
    3. Use Brent's method on the found bracket.
    4. Fallback to global scan if local search fails.
    """
    counter = [0]
    x0 = float(x0)
    
    # Phase 1: Estimate wavelength
    est_step = _estimate_wavelength(f, x0, counter)
    
    # Phase 2: Local search with estimated step
    bracket = _local_search(f, x0, est_step, counter)
    
    if bracket is not None:
        a, b = bracket
        try:
            result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
            if abs(f(result)) <= 1e-8:
                return result
        except Exception:
            pass
    
    # Phase 3: Fallback - try a few other step sizes if the first one failed
    # This handles cases where the wavelength estimate was off.
    fallback_steps = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    for step in fallback_steps:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
            
        bracket = _local_search(f, x0, step, counter)
        if bracket is not None:
            a, b = bracket
            try:
                result = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
                if abs(f(result)) <= 1e-8:
                    return result
            except Exception:
                pass
                
    # Phase 4: Last resort - return x0 if nothing else works
    # This is better than returning None or raising an exception.
    return x0
