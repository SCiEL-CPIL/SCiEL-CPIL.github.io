"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

# Deadline guard: 150s wall-clock limit, use 140s as safety margin
_DEADLINE = time.time() + 140.0

def _time_left():
    return _DEADLINE - time.time()

def find_root(f, x0):
    """
    Hybrid root finder combining secant method, Newton's method, and bisection
    with bracketing. Designed for smooth polynomial equations.
    
    Strategy:
    1. Start with secant method (uses 2 initial points)
    2. If secant diverges or stalls, try to establish a bracket
    3. Use bisection within bracket for guaranteed convergence
    4. Polish with Newton/secant once close
    5. Time guard to return best result before deadline
    """
    x = float(x0)
    
    # Initial evaluations
    fx = f(x)
    if abs(fx) <= 1e-10:
        return x
    if not math.isfinite(fx):
        return x
    
    # Try to get a second point for secant
    # Use a small step relative to x
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = f(x2)
    if not math.isfinite(fx2):
        x2 = x - h
        fx2 = f(x2)
        if not math.isfinite(fx2):
            # Fallback: pure Newton with finite difference
            return _newton_fallback(f, x, fx)
    
    # If we have a sign change between x and x2, use bisection
    if fx * fx2 < 0:
        return _bisection(f, x, fx, x2, fx2)
    
    # No bracket; use secant method with divergence protection
    a, fa = x, fx
    b, fb = x2, fx2
    
    # Track best point (smallest |f|)
    best_x = a if abs(fa) < abs(fb) else b
    best_fx = min(abs(fa), abs(fb))
    
    # Secant iterations
    max_iter = 40
    for i in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Compute secant step
        if fb == fa:
            # Degenerate; try Newton
            h = 1e-8 * max(1.0, abs(b))
            fb2 = f(b + h)
            if math.isfinite(fb2):
                d = (fb2 - fb) / h
                if d != 0 and math.isfinite(d):
                    new_x = b - fb / d
                    if math.isfinite(new_x):
                        new_fx = f(new_x)
                        if math.isfinite(new_fx):
                            if abs(new_fx) < best_fx:
                                best_x = new_x
                                best_fx = abs(new_fx)
                            b, fb = new_x, new_fx
                            a, fa = b, fb  # Reset for next secant
                            continue
            break
        else:
            denom = fb - fa
            new_x = b - fb * (b - a) / denom
        
        if not math.isfinite(new_x):
            break
        
        # Divergence check: if new_x is too far, fall back to bisection-like step
        step = abs(new_x - b)
        if step > 10 * max(1.0, abs(b)):
            # Take a smaller step toward a
            new_x = b + 0.5 * (a - b)
        
        new_fx = f(new_x)
        if not math.isfinite(new_fx):
            # Try a different point
            new_x = 0.5 * (a + b)
            new_fx = f(new_x)
            if not math.isfinite(new_fx):
                break
        
        # Update best
        if abs(new_fx) < best_fx:
            best_x = new_x
            best_fx = abs(new_fx)
        
        if abs(new_fx) <= 1e-12:
            return new_x
        
        # Check for bracket
        if fa * new_fx < 0:
            return _bisection(f, a, fa, new_x, new_fx)
        if fb * new_fx < 0:
            return _bisection(f, b, fb, new_x, new_fx)
        
        # Shift window
        a, fa = b, fb
        b, fb = new_x, new_fx
    
    # If we got close enough, return
    if best_fx <= 1e-8:
        return best_x
    
    # Try to establish a bracket by expanding
    bracket = _find_bracket(f, best_x, best_fx)
    if bracket is not None:
        lo, f_lo, hi, f_hi = bracket
        return _bisection(f, lo, f_lo, hi, f_hi)
    
    # Final polish attempt with Newton
    result = _newton_polish(f, best_x)
    if result is not None and math.isfinite(result):
        return result
    
    return best_x

def _newton_fallback(f, x, fx):
    """Simple Newton with finite difference derivative."""
    for _ in range(30):
        if _time_left() < 5.0:
            break
        if abs(fx) <= 1e-10:
            return x
        h = 1e-7 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        new_x = x - fx / d
        if not math.isfinite(new_x):
            break
        new_fx = f(new_x)
        if not math.isfinite(new_fx):
            break
        x, fx = new_x, new_fx
    return x

def _bisection(f, a, fa, b, fb):
    """Bisection method, guaranteed to converge if fa*fb < 0."""
    if fa * fb > 0:
        # No bracket; return the better point
        return a if abs(fa) < abs(fb) else b
    
    for _ in range(60):
        if _time_left() < 5.0:
            break
        mid = 0.5 * (a + b)
        if mid == a or mid == b:
            break
        fm = f(mid)
        if not math.isfinite(fm):
            break
        if abs(fm) <= 1e-12:
            return mid
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
        if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
            break
    
    # Return the point with smaller |f|
    return a if abs(fa) < abs(fb) else b

def _find_bracket(f, x0, fx0):
    """Try to find a bracket around x0 by expanding."""
    if not math.isfinite(fx0):
        return None
    
    # Try several directions and scales
    scales = [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    for s in scales:
        if _time_left() < 5.0:
            break
        for direction in [1, -1]:
            x1 = x0 + direction * s * max(1.0, abs(x0))
            fx1 = f(x1)
            if not math.isfinite(fx1):
                continue
            if fx0 * fx1 < 0:
                return (x0, fx0, x1, fx1)
    
    return None

def _newton_polish(f, x):
    """High-precision Newton polish."""
    fx = f(x)
    if abs(fx) <= 1e-10:
        return x
    for _ in range(15):
        if _time_left() < 3.0:
            break
        h = 1e-9 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        new_x = x - fx / d
        if not math.isfinite(new_x):
            break
        new_fx = f(new_x)
        if not math.isfinite(new_fx):
            break
        if abs(new_fx) > abs(fx):
            break  # Diverging
        x, fx = new_x, new_fx
        if abs(fx) <= 1e-12:
            break
    return x if abs(fx) <= 1e-8 else None
# EVOLVE-BLOCK-END
