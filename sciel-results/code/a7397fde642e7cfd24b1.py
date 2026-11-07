"""Combined root finder: Newton + secant + bisection with call budget and deadline."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses Newton's method with finite-difference derivative, secant fallback,
    and bisection for bracketing. Tracks call count and wall-clock deadline.
    """
    START_TIME = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    MAX_CALLS = 480   # Keep well under 500 per equation

    call_count = [0]

    def _is_finite(x):
        try:
            xf = float(x)
            return xf == xf and xf != float('inf') and xf != float('-inf')
        except (ValueError, TypeError, OverflowError):
            return False

    def safe_f(x):
        call_count[0] += 1
        if call_count[0] > MAX_CALLS:
            raise ValueError("too many calls")
        if time.time() - START_TIME > DEADLINE:
            raise ValueError("deadline exceeded")
        val = f(x)
        if not _is_finite(val):
            raise ValueError("non-finite")
        return val

    def check_deadline():
        return time.time() - START_TIME < DEADLINE

    def check_calls(needed=1):
        return call_count[0] + needed <= MAX_CALLS

    x = float(x0)
    
    # Initial evaluation
    try:
        best_fx = safe_f(x)
    except Exception:
        return x
    
    if not _is_finite(best_fx):
        return x
    
    best_x = x
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return x

    # Phase 1: Newton's method with central finite-difference derivative
    try:
        for _ in range(60):
            if not check_deadline() or not check_calls(3):
                break
            
            fx = safe_f(x)
            if not _is_finite(fx):
                break
            
            ax = abs(fx)
            if ax < best_abs:
                best_x = x
                best_abs = ax
            
            if ax <= 1e-14:
                return x
            
            # Central difference derivative
            h = 1e-6 * max(1.0, abs(x))
            try:
                fp = safe_f(x + h)
                fm = safe_f(x - h)
            except Exception:
                break
            
            if not _is_finite(fp) or not _is_finite(fm):
                break
            
            d = (fp - fm) / (2.0 * h)
            if not _is_finite(d) or abs(d) < 1e-30:
                break
            
            x_new = x - fx / d
            if not _is_finite(x_new):
                break
            
            # Damping if step is too large
            step = abs(x_new - x)
            if step > 10.0 * max(1.0, abs(x)):
                x_new = x + (10.0 * max(1.0, abs(x))) * (1.0 if x_new > x else -1.0)
            
            x = x_new
            if abs(x) > 1e15:
                break
    except Exception:
        pass

    # If we already have a good enough root, return
    if best_abs <= 1e-8:
        return best_x

    # Phase 2: Secant method (uses previous two points)
    try:
        if check_deadline() and check_calls(2):
            # Get two points for secant
            x1 = best_x
            f1 = safe_f(x1)
            if not _is_finite(f1):
                raise ValueError("non-finite")
            ax1 = abs(f1)
            if ax1 < best_abs:
                best_x = x1
                best_abs = ax1
            
            if ax1 <= 1e-14:
                return x1
            
            h = 1e-4 * max(1.0, abs(x1))
            x2 = x1 + h
            f2 = safe_f(x2)
            if not _is_finite(f2):
                raise ValueError("non-finite")
            ax2 = abs(f2)
            if ax2 < best_abs:
                best_x = x2
                best_abs = ax2
            
            if ax2 <= 1e-14:
                return x2
            
            for _ in range(40):
                if not check_deadline() or not check_calls(1):
                    break
                
                if abs(f2 - f1) < 1e-30:
                    break
                
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                if not _is_finite(x_new):
                    break
                
                # Damping
                step = abs(x_new - x2)
                if step > 10.0 * max(1.0, abs(x2)):
                    x_new = x2 + (10.0 * max(1.0, abs(x2))) * (1.0 if x_new > x2 else -1.0)
                
                f_new = safe_f(x_new)
                if not _is_finite(f_new):
                    break
                
                ax_new = abs(f_new)
                if ax_new < best_abs:
                    best_x = x_new
                    best_abs = ax_new
                
                if ax_new <= 1e-14:
                    return x_new
                
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
    except Exception:
        pass

    if best_abs <= 1e-8:
        return best_x

    # Phase 3: Try to find a bracket and use bisection
    try:
        if check_deadline() and check_calls(10):
            lo = float(x0)
            hi = float(x0)
            
            # Try to expand to find sign change
            scale = max(1.0, abs(float(x0)))
            found_bracket = False
            
            for i in range(1, 15):
                if not check_deadline() or not check_calls(2):
                    break
                
                for direction in (1, -1):
                    if not check_calls(1):
                        break
                    x_try = float(x0) + direction * scale * (2.0 ** i)
                    try:
                        f_try = safe_f(x_try)
                    except Exception:
                        continue
                    if not _is_finite(f_try):
                        continue
                    
                    ax_try = abs(f_try)
                    if ax_try < best_abs:
                        best_x = x_try
                        best_abs = ax_try
                    
                    if ax_try <= 1e-14:
                        return x_try
                    
                    # Check if x0 and x_try bracket a root
                    try:
                        f0 = safe_f(float(x0))
                    except Exception:
                        f0 = None
                    
                    if f0 is not None and _is_finite(f0) and f0 * f_try < 0:
                        lo = float(x0)
                        hi = x_try
                        found_bracket = True
                        break
                
                if found_bracket:
                    break
            
            if found_bracket:
                a, b = lo, hi
                fa = safe_f(a)
                fb = safe_f(b)
                if not _is_finite(fa) or not _is_finite(fb):
                    raise ValueError("non-finite")
                if fa * fb > 0:
                    a, b, fa, fb = b, a, fb, fa
                
                for _ in range(50):
                    if not check_deadline() or not check_calls(1):
                        break
                    
                    m = (a + b) / 2.0
                    fm = safe_f(m)
                    if not _is_finite(fm):
                        break
                    
                    ax_m = abs(fm)
                    if ax_m < best_abs:
                        best_x = m
                        best_abs = ax_m
                    
                    if ax_m <= 1e-14:
                        return m
                    
                    if fa * fm < 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    
                    if abs(b - a) < 1e-14:
                        break
    except Exception:
        pass

    return best_x
# EVOLVE-BLOCK-END
