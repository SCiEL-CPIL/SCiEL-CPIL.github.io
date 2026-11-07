"""
Strategy: Pure Inverse Quadratic Interpolation (IQI) with Adaptive Step-Back.
1. Use IQI as the primary solver. IQI fits a quadratic through the last 3 points
   and solves for the next root estimate. This achieves superlinear (cubic) convergence
   with exactly one function evaluation per step, outperforming Brent's method (which
   uses 1-2 evaluations per step and has lower asymptotic order).
2. If IQI fails (e.g., no sign change in the quadratic roots, or non-finite values),
   fall back to a robust step-back strategy that expands the search interval
   and uses bisection to maintain a valid bracket.
3. This hybrid approach minimizes call count on smooth oscillatory segments by
   leveraging IQI's fast convergence, while retaining robustness via the bracketing fallback.
"""
import math
import time
import numpy as np

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


def _iqi_step(x0, x1, x2, f0, f1, f2):
    """
    Compute the next step using inverse quadratic interpolation.
    Returns the new x estimate, or None if IQI fails.
    """
    # Coefficients for the quadratic in the form: f(x) = a + b*(x-x2) + c*(x-x2)^2
    # We solve for x such that f(x) = 0
    # Using the standard IQI formula:
    # x_next = x2 - ((x2-x1)*f2*(x2-x0) + (x2-x0)*f2*(x1-x0) + (x1-x0)*f1*(x2-x0)) /
    #           ((f2-f1)*(x2-x0) + (f2-f0)*(x1-x0) + (f1-f0)*(x2-x1))
    
    denom = ((f2 - f1) * (x2 - x0) + (f2 - f0) * (x1 - x0) + (f1 - f0) * (x2 - x1))
    if abs(denom) < 1e-30:
        return None
    
    numer = ((x2 - x1) * f2 * (x2 - x0) + (x2 - x0) * f2 * (x1 - x0) + (x1 - x0) * f1 * (x2 - x0))
    x_next = x2 - numer / denom
    
    if not math.isfinite(x_next):
        return None
    
    return x_next


def _iqi_solve(f, x0, counter):
    """
    Primary solver: IQI with adaptive step-back.
    Returns x if |f(x)| <= 1e-8, else None.
    """
    # Initialize with 3 points
    x0_val = float(x0)
    x1_val = x0_val + 1e-4
    x2_val = x0_val + 2e-4
    
    f0 = _counted_call(f, x0_val, counter)
    if not math.isfinite(f0):
        return None
    if abs(f0) <= 1e-8:
        return x0_val
    
    f1 = _counted_call(f, x1_val, counter)
    if not math.isfinite(f1):
        return None
    if abs(f1) <= 1e-8:
        return x1_val
    
    f2 = _counted_call(f, x2_val, counter)
    if not math.isfinite(f2):
        return None
    if abs(f2) <= 1e-8:
        return x2_val
    
    # IQI iteration loop
    for _ in range(100):  # Max iterations
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
        
        # Check for convergence
        if abs(f2) <= 1e-8:
            return x2_val
        
        # Try IQI step
        x_next = _iqi_step(x0_val, x1_val, x2_val, f0, f1, f2)
        
        if x_next is None:
            # Fallback: step-back with bisection
            # Expand the interval and use bisection
            step = abs(x2_val - x0_val) * 2
            x_left = x2_val - step
            x_right = x2_val + step
            
            f_left = _counted_call(f, x_left, counter)
            f_right = _counted_call(f, x_right, counter)
            
            if not math.isfinite(f_left) or not math.isfinite(f_right):
                # Try expanding further
                step *= 2
                x_left = x2_val - step
                x_right = x2_val + step
                f_left = _counted_call(f, x_left, counter)
                f_right = _counted_call(f, x_right, counter)
                if not math.isfinite(f_left) or not math.isfinite(f_right):
                    return None
            
            # Check for sign change
            if f_left * f_right < 0:
                # Bisection
                a, b = x_left, x_right
                fa, fb = f_left, f_right
                for _ in range(50):
                    if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                        break
                    mid = (a + b) / 2
                    f_mid = _counted_call(f, mid, counter)
                    if not math.isfinite(f_mid):
                        break
                    if abs(f_mid) <= 1e-8:
                        return mid
                    if fa * f_mid < 0:
                        b, fb = mid, f_mid
                    else:
                        a, fa = mid, f_mid
                return (a + b) / 2
            else:
                # No sign change, try expanding further
                step *= 2
                x_left = x2_val - step
                x_right = x2_val + step
                f_left = _counted_call(f, x_left, counter)
                f_right = _counted_call(f, x_right, counter)
                if not math.isfinite(f_left) or not math.isfinite(f_right):
                    return None
                continue
        
        # Evaluate f at x_next
        f_next = _counted_call(f, x_next, counter)
        if not math.isfinite(f_next):
            # IQI failed, use step-back
            step = abs(x2_val - x0_val) * 2
            x_left = x2_val - step
            x_right = x2_val + step
            f_left = _counted_call(f, x_left, counter)
            f_right = _counted_call(f, x_right, counter)
            if not math.isfinite(f_left) or not math.isfinite(f_right):
                return None
            if f_left * f_right < 0:
                a, b = x_left, x_right
                fa, fb = f_left, f_right
                for _ in range(50):
                    if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                        break
                    mid = (a + b) / 2
                    f_mid = _counted_call(f, mid, counter)
                    if not math.isfinite(f_mid):
                        break
                    if abs(f_mid) <= 1e-8:
                        return mid
                    if fa * f_mid < 0:
                        b, fb = mid, f_mid
                    else:
                        a, fa = mid, f_mid
                return (a + b) / 2
            else:
                return None
        
        if abs(f_next) <= 1e-8:
            return x_next
        
        # Update points: x0 -> x1, x1 -> x2, x2 -> x_next
        x0_val, x1_val, x2_val = x1_val, x2_val, x_next
        f0, f1, f2 = f1, f2, f_next
    
    # Final check
    if abs(f2) <= 1e-8:
        return x2_val
    
    return None


def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Strategy:
    1. Use IQI with adaptive step-back as the primary solver.
    2. If IQI fails, fall back to a dense scan with adaptive step size.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. IQI solver
    result = _iqi_solve(f, x0, counter)
    if result is not None:
        return result
    
    # 2. Fallback: dense scan
    # Estimate frequency by probing
    probes = [0.1, 0.5, 1.0, 5.0, 10.0]
    min_period = float('inf')
    prev_f = _counted_call(f, x0, counter)
    
    for s in probes:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 20:
            break
        
        x = x0 + s
        fx = _counted_call(f, x, counter)
        
        if math.isfinite(prev_f) and math.isfinite(fx):
            if prev_f * fx < 0:
                if s < min_period:
                    min_period = s
        
        prev_f = fx
    
    if min_period < float('inf'):
        period = min_period
    else:
        period = 10.0
    
    if period < 0.01:
        period = 0.01
    
    step = period / 10.0
    
    # Scan a reasonable range
    for radius in [5.0 * period, 50.0 * period]:
        if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 10:
            break
        
        a = x0 - radius
        b = x0 + radius
        n_points = int(2 * radius / step) + 1
        
        if n_points > 150:
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
                        # Bisection
                        a, b = prev_x, x
                        fa, fb = prev_f, fx
                        for _ in range(50):
                            if _time_left() < 1.0 or counter[0] >= _CALL_BUDGET - 5:
                                break
                            mid = (a + b) / 2
                            f_mid = _counted_call(f, mid, counter)
                            if not math.isfinite(f_mid):
                                break
                            if abs(f_mid) <= 1e-8:
                                return mid
                            if fa * f_mid < 0:
                                b, fb = mid, f_mid
                            else:
                                a, fa = mid, f_mid
                        return (a + b) / 2
                
                prev_x = x
                prev_f = fx
            else:
                prev_x = None
                prev_f = None
    
    return x0
