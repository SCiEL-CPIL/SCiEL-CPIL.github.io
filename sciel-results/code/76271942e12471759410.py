"""
Strategy: Adaptive Multi-Scale Frequency Search.
1. Use a small set of probe points to estimate the dominant oscillation frequency.
2. Use the estimated frequency to determine an optimal step size for a dense scan.
3. Perform a targeted scan at that resolution to find a sign-changing bracket.
4. Use scipy.optimize.brentq to converge.
This avoids wasting calls on a fixed grid that may be too coarse for high-frequency 
oscillations or too fine for low-frequency ones.
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


def _estimate_frequency(f, x0, counter):
    """
    Estimate the dominant frequency by evaluating f at a few points 
    and looking for sign changes or zero crossings.
    Returns an estimated period T (or None if uncertain).
    """
    # Probe at various scales to catch different frequencies
    # We want to find the smallest scale where we see a sign change, 
    # which suggests the period is roughly 2*scale or less.
    # But we also need to be careful not to jump over a root.
    
    # Let's probe at scales: 0.1, 0.5, 1.0, 5.0, 10.0
    # If we find a sign change at scale s, the period is likely <= 2*s.
    # If we don't find one at scale s, the period might be > 2*s.
    
    # Actually, a better heuristic:
    # Evaluate f at x0, x0+0.1, x0+0.5, x0+1, x0+5, x0+10.
    # Count sign changes between consecutive points.
    # If we see a sign change between x and x+s, the period is likely ~2s or smaller.
    # If we see no sign changes, the period is likely larger.
    
    probes = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    prev_x = x0
    prev_f = _counted_call(f, x0, counter)
    
    min_period = float('inf')
    
    for s in probes:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 20:
            break
            
        x = x0 + s
        fx = _counted_call(f, x, counter)
        
        if math.isfinite(prev_f) and math.isfinite(fx):
            if prev_f * fx < 0:
                # Sign change detected in interval of length s.
                # This suggests the period is roughly 2*s or less.
                # To be safe, assume period is ~s.
                if s < min_period:
                    min_period = s
                # If we found a sign change at a small scale, we can stop?
                # Not necessarily, but it gives a good upper bound.
                # However, if the function is highly oscillatory, there might be 
                # multiple roots in this interval. 
                # But for the purpose of step size, a smaller step is safer.
        
        prev_x = x
        prev_f = fx
        
    if min_period < float('inf'):
        return min_period
    else:
        # No sign changes found in probes. 
        # Assume a default period, e.g., 10.0? 
        # Or return None to trigger a fallback.
        return 10.0


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy: 
    1. Check x0 directly.
    2. Estimate frequency/period.
    3. Perform a dense scan based on the estimated period.
    4. Use scipy.optimize.brentq to converge within the bracket.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Check x0
    f0 = _counted_call(f, x0, counter)
    if math.isfinite(f0) and abs(f0) <= 1e-8:
        return x0

    # 2. Estimate period
    period = _estimate_frequency(f, x0, counter)
    
    # If period is very small, we need a very fine step.
    # If period is large, we can use a coarser step.
    # Let's aim for a step size of period / 10 to ensure we catch sign changes.
    # But we also need to cover a reasonable range.
    
    # If period is None or very large, use a default.
    if period is None or period > 100:
        period = 10.0
    if period < 0.01:
        period = 0.01
        
    step = period / 10.0
    
    # 3. Dense scan
    # We will scan a range around x0. 
    # How wide? Let's try a few ranges.
    # Range 1: x0 +/- 5*period
    # Range 2: x0 +/- 50*period
    # Range 3: x0 +/- 500*period
    
    ranges = [5.0 * period, 50.0 * period, 500.0 * period]
    
    for radius in ranges:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
            
        a = x0 - radius
        b = x0 + radius
        n_points = int(2 * radius / step) + 1
        
        # Cap the number of points to avoid excessive calls in one scan
        if n_points > 150:
            # Adjust step to fit 150 points
            step = 2 * radius / 149
            n_points = 150
            
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
                        # Bracket found: [prev_x, x]
                        try:
                            result = brentq(f, prev_x, x, xtol=1e-12, rtol=1e-12, maxiter=100)
                            if abs(f(result)) <= 1e-8:
                                return result
                        except Exception:
                            pass
                
                prev_x = x
                prev_f = fx
            else:
                # If fx is not finite, reset previous state
                prev_x = None
                prev_f = None

    # Fallback: If we didn't find a bracket, try a random search?
    # Or just return x0.
    return x0
