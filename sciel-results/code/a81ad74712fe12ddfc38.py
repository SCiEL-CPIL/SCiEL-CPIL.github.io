"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 140.0  # safety margin before 150s limit
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS:
            return None
        if time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isnan(val) or math.isinf(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
        return val
    
    def is_good(val):
        return val is not None and abs(val) <= TOL
    
    def newton_raphson(x0, max_iter=80):
        nonlocal evals, best_x, best_abs
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return best_x
        if is_good(fx):
            return x
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE:
                return best_x
            # Adaptive step size for derivative
            h = 1e-8 * max(1.0, abs(x))
            # Try central difference if possible, otherwise forward
            xp = x + h
            xm = x - h
            fpx = call_f(xp)
            fxm = call_f(xm)
            
            if fpx is not None and fxm is not None:
                d = (fpx - fxm) / (2 * h)
            elif fpx is not None:
                d = (fpx - fx) / h
            elif fxm is not None:
                d = (fx - fxm) / h
            else:
                break
            
            if d == 0 or math.isnan(d) or math.isinf(d):
                break
            
            x_new = x - fx / d
            
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
            
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            
            if is_good(fx_new):
                return x_new
            
            # Check for convergence
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                break
            
            x = x_new
            fx = fx_new
        
        return best_x
    
    def secant(x0, x1, max_iter=80):
        nonlocal evals, best_x, best_abs
        x_prev = float(x0)
        x_curr = float(x1)
        f_prev = call_f(x_prev)
        f_curr = call_f(x_curr)
        
        if f_prev is None or f_curr is None:
            return best_x
        if is_good(f_prev):
            return x_prev
        if is_good(f_curr):
            return x_curr
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE:
                return best_x
            
            if f_curr - f_prev == 0:
                break
            
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            # Damping
            if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_new = x_curr + 0.5 * (x_new - x_curr)
            
            f_new = call_f(x_new)
            if f_new is None:
                break
            
            if is_good(f_new):
                return x_new
            
            if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_new)):
                break
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
        
        return best_x
    
    def bisection(xa, xb, fa, fb, max_iter=100):
        nonlocal evals, best_x, best_abs
        if fa is None or fb is None:
            return best_x
        if is_good(fa):
            return xa
        if is_good(fb):
            return xb
        
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            return best_x
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE:
                return best_x
            
            x_mid = (xa + xb) / 2.0
            f_mid = call_f(x_mid)
            if f_mid is None:
                break
            
            if is_good(f_mid):
                return x_mid
            
            if f_mid * fa <= 0:
                xb = x_mid
                fb = f_mid
            else:
                xa = x_mid
                fa = f_mid
            
            if abs(xb - xa) < 1e-14 * max(1.0, abs(x_mid)):
                break
        
        return best_x
    
    # Strategy: Try Newton first, then secant, then bisection with bracketing
    x0 = float(x0)
    
    # Phase 1: Newton's method
    result = newton_raphson(x0, max_iter=50)
    if best_abs <= TOL:
        return best_x
    
    # Phase 2: Secant method with perturbed starting points
    h = 1e-4 * max(1.0, abs(x0))
    result = secant(x0, x0 + h, max_iter=50)
    if best_abs <= TOL:
        return best_x
    
    result = secant(x0 - h, x0 + h, max_iter=50)
    if best_abs <= TOL:
        return best_x
    
    # Phase 3: Try to find a bracket and use bisection
    # Expand search interval
    step = 1e-2 * max(1.0, abs(x0))
    for i in range(1, 20):
        if time.monotonic() > DEADLINE:
            break
        if evals >= MAX_EVALS:
            break
        
        xa = x0 - i * step
        xb = x0 + i * step
        fa = call_f(xa)
        fb = call_f(xb)
        
        if fa is None or fb is None:
            continue
        
        if is_good(fa):
            return xa
        if is_good(fb):
            return xb
        
        if fa * fb < 0:
            result = bisection(xa, xb, fa, fb, max_iter=60)
            if best_abs <= TOL:
                return best_x
    
    return best_x
# EVOLVE-BLOCK-END
