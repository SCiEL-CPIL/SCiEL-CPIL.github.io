"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses a hybrid approach:
    1. Try Newton's method with damping and bracket tracking
    2. Fall back to bisection if bracket is found
    3. Use secant method if no bracket
    4. Multiple restart strategies if initial guess fails
    """
    deadline = time.time() + 140.0  # Leave 10s margin before 150s kill
    max_evals = 450
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not np.isfinite(val):
                return None
            return float(val)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return None
    
    x0 = float(x0)
    
    # Try to get an initial bracket
    def find_bracket(x0, f_eval):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        a = x0
        fa = f_eval(a)
        if fa is None:
            return None
        if abs(fa) <= 1e-12:
            return (a, a)
        
        # Try expanding in both directions
        step = max(1e-6, abs(a) * 1e-6)
        for i in range(100):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            
            # Try positive direction
            b = a + step
            fb = f_eval(b)
            if fb is not None:
                if fa * fb <= 0:
                    return (a, b)
                # Check if b is closer to root
                if abs(fb) < abs(fa):
                    a, fa = b, fb
                    step *= 2
                    continue
            step *= 2
            if step > 1e10:
                break
        
        # Try negative direction
        a = x0
        fa = f_eval(a)
        if fa is None:
            return None
        if abs(fa) <= 1e-12:
            return (a, a)
        
        step = max(1e-6, abs(a) * 1e-6)
        for i in range(100):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            
            b = a - step
            fb = f_eval(b)
            if fb is not None:
                if fa * fb <= 0:
                    return (b, a)
                if abs(fb) < abs(fa):
                    a, fa = b, fb
                    step *= 2
                    continue
            step *= 2
            if step > 1e10:
                break
        
        return None
    
    # Bisection method
    def bisection(a, b, fa, fb):
        for _ in range(200):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                return mid
        return (a + b) / 2.0
    
    # Newton's method with damping
    def newton(x):
        for _ in range(100):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            fx = eval_f(x)
            if fx is None:
                break
            if abs(fx) <= 1e-12:
                return x
            
            # Compute derivative using central difference
            h = 1e-7 * max(1.0, abs(x))
            fxp = eval_f(x + h)
            fxm = eval_f(x - h)
            if fxp is None or fxm is None:
                break
            dfx = (fxp - fxm) / (2.0 * h)
            if dfx == 0 or not np.isfinite(dfx):
                break
            
            # Newton step with damping
            step = fx / dfx
            if not np.isfinite(step):
                break
            
            # Damp if step is too large
            if abs(step) > abs(x) * 0.5:
                step = 0.5 * abs(x) * (1.0 if step > 0 else -1.0)
            
            x_new = x - step
            
            # Check for convergence
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                return x_new
            
            # Check if we're making progress
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) < abs(fx) * 0.5:
                x = x_new
            else:
                # Halve the step
                x = x - step / 2.0
                fx = eval_f(x)
                if fx is None:
                    break
                if abs(fx) <= 1e-12:
                    return x
        
        return x
    
    # Secant method
    def secant(x0, x1):
        fx0 = eval_f(x0)
        if fx0 is None:
            return None
        if abs(fx0) <= 1e-12:
            return x0
        
        fx1 = eval_f(x1)
        if fx1 is None:
            return None
        if abs(fx1) <= 1e-12:
            return x1
        
        for _ in range(100):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            if fx1 == fx0:
                break
            slope = (fx1 - fx0) / (x1 - x0)
            if slope == 0 or not np.isfinite(slope):
                break
            x2 = x1 - fx1 / slope
            if not np.isfinite(x2):
                break
            fx2 = eval_f(x2)
            if fx2 is None:
                break
            if abs(fx2) <= 1e-12:
                return x2
            if abs(x2 - x1) < 1e-14 * max(1.0, abs(x2)):
                return x2
            x0, fx0 = x1, fx1
            x1, fx1 = x2, fx2
        return x1
    
    # Strategy 1: Try to find a bracket and use bisection
    bracket = find_bracket(x0, eval_f)
    if bracket is not None:
        a, b = bracket
        if a == b:
            return a
        fa = eval_f(a)
        fb = eval_f(b)
        if fa is not None and fb is not None:
            result = bisection(a, b, fa, fb)
            if result is not None:
                fr = eval_f(result)
                if fr is not None and abs(fr) <= 1e-8:
                    return result
    
    # Strategy 2: Try Newton's method
    result = newton(x0)
    if result is not None:
        fr = eval_f(result)
        if fr is not None and abs(fr) <= 1e-8:
            return result
    
    # Strategy 3: Try secant method with perturbed starting points
    h = max(1e-6, abs(x0) * 1e-6)
    for dx in [h, -h, 2*h, -2*h, 10*h, -10*h]:
        if time.time() > deadline or eval_count[0] >= max_evals:
            break
        x1 = x0 + dx
        result = secant(x0, x1)
        if result is not None:
            fr = eval_f(result)
            if fr is not None and abs(fr) <= 1e-8:
                return result
    
    # Strategy 4: Try multiple restarts with different initial guesses
    for scale in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        if time.time() > deadline or eval_count[0] >= max_evals:
            break
        for sign in [1.0, -1.0]:
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            x_try = x0 + sign * scale * max(1.0, abs(x0))
            if x_try == x0:
                continue
            result = newton(x_try)
            if result is not None:
                fr = eval_f(result)
                if fr is not None and abs(fr) <= 1e-8:
                    return result
    
    # Return best result found (x0 if nothing else worked)
    return x0
# EVOLVE-BLOCK-END
