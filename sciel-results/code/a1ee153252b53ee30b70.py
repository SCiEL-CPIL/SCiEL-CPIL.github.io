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
    
    def bisection_secant(a, fa, b, fb, max_iters=50):
        """Hybrid bisection + secant. Returns root or None."""
        if time.monotonic() > DEADLINE:
            return None
        if fa is None or fb is None:
            return None
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            return None
        
        x_prev = a
        f_prev = fa
        x_curr = b
        f_curr = fb
        
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if evals >= MAX_EVALS - 2:
                break
            
            # Try secant step
            if f_curr != f_prev:
                x_sec = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                # Check if secant step stays within bracket
                if (x_sec - a) * (x_sec - b) <= 0 and math.isfinite(x_sec):
                    x_new = x_sec
                else:
                    # Bisection
                    x_new = (a + b) / 2.0
            else:
                x_new = (a + b) / 2.0
            
            f_new = call_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new
            
            # Update bracket
            if fa * f_new <= 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            
            # Converged by bracket size
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                # Return midpoint
                x_mid = (a + b) / 2.0
                if time.monotonic() < DEADLINE and evals < MAX_EVALS:
                    f_mid = call_f(x_mid)
                    if f_mid is not None and abs(f_mid) <= TOL:
                        return x_mid
                return x_mid if best_abs <= TOL else None
        
        return None
    
    def secant(x0, max_iters=40):
        """Pure secant method."""
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
            if evals >= MAX_EVALS - 1:
                break
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
    
    def newton_raphson(x0, max_iters=30):
        """Newton with numerical derivative."""
        x = float(x0)
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if evals >= MAX_EVALS - 1:
                break
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
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
            x = x_new
        return None
    
    # Strategy:
    # 1. Evaluate f(x0)
    # 2. Try to find a bracket by sampling nearby points
    # 3. If bracket found, use bisection_secant
    # 4. Otherwise, try secant and Newton
    
    fx0 = call_f(x0)
    if fx0 is not None and abs(fx0) <= TOL:
        return x0
    
    # Try to find a sign change bracket around x0
    bracket_a = None
    bracket_b = None
    bracket_fa = None
    bracket_fb = None
    
    if fx0 is not None:
        # Sample at various distances to find sign change
        for dist in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]:
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
                break
            x_try = float(x0) + dist
            f_try = call_f(x_try)
            if f_try is None:
                continue
            if fx0 * f_try < 0:
                # Found sign change
                if abs(fx0) < abs(f_try):
                    bracket_a, bracket_b = x0, x_try
                    bracket_fa, bracket_fb = fx0, f_try
                else:
                    bracket_a, bracket_b = x_try, x0
                    bracket_fa, bracket_fb = f_try, fx0
                break
    
    # If bracket found, use bisection_secant
    if bracket_a is not None:
        result = bisection_secant(bracket_a, bracket_fa, bracket_b, bracket_fb, max_iters=60)
        if result is not None:
            return result
    
    # Try secant from x0
    result = secant(x0, max_iters=40)
    if result is not None:
        return result
    
    # Try Newton from x0
    result = newton_raphson(x0, max_iters=30)
    if result is not None:
        return result
    
    # Try secant and Newton from offset points
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
            break
        x_try = float(x0) + offset
        result = secant(x_try, max_iters=20)
        if result is not None:
            return result
        result = newton_raphson(x_try, max_iters=15)
        if result is not None:
            return result
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
