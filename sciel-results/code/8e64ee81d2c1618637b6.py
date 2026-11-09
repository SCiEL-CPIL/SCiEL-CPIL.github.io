"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of:
    1. Newton's method with adaptive step size
    2. Bisection fallback for robustness
    3. Secant method when derivative is unreliable
    
    Call budget: < 500 calls per equation
    """
    x = float(x0)
    best_x = x
    best_fx = None
    
    # Count calls (for internal tracking, not returned)
    call_count = 0
    
    def counted_eval(x_val):
        nonlocal call_count, best_x, best_fx
        if not _is_finite(x_val):
            return None
        val = f(x_val)
        call_count += 1
        if val is not None and isinstance(val, (int, float)) and math.isfinite(val):
            if best_fx is None or abs(val) < abs(best_fx):
                best_x = x_val
                best_fx = val
            return float(val)
        return None
    
    # Initial evaluation
    fx = counted_eval(x)
    if fx is None:
        # Try a small perturbation
        x = x0 + 1e-6
        fx = counted_eval(x)
        if fx is None:
            return x0
    if abs(fx) <= 1e-8:
        return x
    
    # Check deadline
    if _time_left() <= 0:
        return best_x
    
    # Newton's method with adaptive step and bisection fallback
    # Maintain a bracket [lo, hi] where f(lo) and f(hi) have opposite signs
    lo = x - 1.0
    hi = x + 1.0
    f_lo = counted_eval(lo)
    f_hi = counted_eval(hi)
    
    # Try to establish a bracket
    if f_lo is not None and f_hi is not None:
        if f_lo * f_hi < 0:
            pass  # Good bracket
        else:
            # Expand bracket
            for _ in range(20):
                if _time_left() <= 0:
                    return best_x
                lo = x - 2.0 * (hi - x)
                hi = x + 2.0 * (hi - x)
                f_lo = counted_eval(lo)
                f_hi = counted_eval(hi)
                if f_lo is None or f_hi is None:
                    break
                if f_lo * f_hi < 0:
                    break
            else:
                # No bracket found, just use Newton
                pass
    else:
        lo = None
        hi = None
        f_lo = None
        f_hi = None
    
    # Main iteration loop
    for iteration in range(80):
        if _time_left() <= 0:
            return best_x
        
        # Try Newton step
        h = 1e-8 * max(1.0, abs(x))
        f_xh = counted_eval(x + h)
        if f_xh is None:
            break
        
        d = (f_xh - fx) / h if h != 0 else 0
        
        newton_x = None
        if d != 0 and abs(d) > 1e-15:
            newton_x = x - fx / d
        
        # Decide which step to take
        if newton_x is not None and _is_finite(newton_x):
            # Check if Newton step stays within bracket
            if lo is not None and hi is not None:
                if (lo < newton_x < hi) or (hi < newton_x < lo):
                    # Newton step is within bracket, take it
                    x_new = newton_x
                else:
                    # Newton step is outside bracket, use bisection
                    x_new = (lo + hi) / 2.0
            else:
                # No bracket, use Newton with damping
                step = x_new - x
                if abs(step) > 10 * max(1.0, abs(x)):
                    x_new = x + 0.5 * math.copysign(1.0, step) * min(abs(step), 10 * max(1.0, abs(x)))
                else:
                    x_new = x_new
        else:
            # Derivative is zero or unreliable, use secant or bisection
            if lo is not None and hi is not None:
                x_new = (lo + hi) / 2.0
            else:
                # Secant-like step
                if f_xh is not None and f_xh != fx:
                    x_new = x - fx * h / (f_xh - fx)
                else:
                    x_new = x + 1e-4
                    if not _is_finite(x_new):
                        break
        
        if not _is_finite(x_new):
            break
        
        # Evaluate at new point
        fx_new = counted_eval(x_new)
        if fx_new is None:
            break
        
        # Update bracket if we have one
        if lo is not None and hi is not None and f_lo is not None and f_hi is not None:
            if f_lo * fx_new < 0:
                hi = x
                f_hi = fx
            elif f_hi * fx_new < 0:
                lo = x
                f_lo = fx
            else:
                # No sign change, shrink bracket toward x
                if abs(fx_new) < abs(f_lo):
                    lo = x
                    f_lo = fx
                else:
                    hi = x
                    f_hi = fx
        
        x = x_new
        fx = fx_new
        
        if abs(fx) <= 1e-8:
            return x
        
        # Convergence check
        if abs(x - best_x) < 1e-15:
            break
    
    return best_x
