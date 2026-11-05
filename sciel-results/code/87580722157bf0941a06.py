"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val
    
    def is_good(x, fx):
        return fx is not None and math.isfinite(fx) and abs(fx) <= TOL
    
    def newton_raphson(x0, max_iters=40):
        x = float(x0)
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            fx = call_f(x)
            if fx is None:
                return None
            if abs(fx) <= TOL:
                return x
            h = 1e-7 * max(1.0, abs(x))
            fx2 = call_f(x + h)
            if fx2 is None:
                return None
            d = (fx2 - fx) / h
            if d == 0 or not math.isfinite(d):
                return None
            x_new = x - fx / d
            if not math.isfinite(x_new):
                return None
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
            x = x_new
        return None
    
    def secant(x0, max_iters=40):
        x0 = float(x0)
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        f0 = call_f(x0)
        f1 = call_f(x1)
        if f0 is None or f1 is None:
            return None
        if abs(f0) <= TOL:
            return x0
        if abs(f1) <= TOL:
            return x1
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not math.isfinite(x2):
                break
            f2 = call_f(x2)
            if f2 is None:
                break
            if abs(f2) <= TOL:
                return x2
            x0, x1 = x1, x2
            f0, f1 = f1, f2
        return None
    
    # Try Newton first
    result = newton_raphson(x0)
    if result is not None:
        return result
    
    # Try secant
    result = secant(x0)
    if result is not None:
        return result
    
    # Try Newton with different initial points
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if time.monotonic() > DEADLINE:
            break
        x_try = float(x0) + offset
        result = newton_raphson(x_try, max_iters=20)
        if result is not None:
            return result
        result = secant(x_try, max_iters=20)
        if result is not None:
            return result
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
