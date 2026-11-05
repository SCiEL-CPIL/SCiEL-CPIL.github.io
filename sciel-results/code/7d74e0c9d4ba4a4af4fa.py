"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from scipy.optimize import brentq

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
    
    def find_bracket(x0, max_expansions=30):
        """
        Find a bracket [a, b] where f(a) and f(b) have opposite signs.
        Uses geometric expansion in both directions.
        Returns (a, b, fa, fb) or None if no bracket found.
        """
        nonlocal evals, best_x, best_abs
        
        x0 = float(x0)
        fx0 = call_f(x0)
        if fx0 is None:
            return None
        if is_good(fx0):
            return None  # Already have a root
        
        # Try expanding in both directions
        base_step = 1e-4 * max(1.0, abs(x0))
        
        # First try small steps around x0
        for i in range(1, 10):
            if evals >= MAX_EVALS - 50 or time.monotonic() > DEADLINE:
                break
            step = base_step * (i ** 2)  # Quadratic growth
            xa = x0 - step
            xb = x0 + step
            fa = call_f(xa)
            fb = call_f(xb)
            
            if fa is not None and is_good(fa):
                return None
            if fb is not None and is_good(fb):
                return None
            
            if fa is not None and fb is not None and fa * fb < 0:
                return (xa, xb, fa, fb)
        
        # Larger expansions
        for i in range(1, max_expansions):
            if evals >= MAX_EVALS - 50 or time.monotonic() > DEADLINE:
                break
            step = base_step * (10 ** (i / 3.0))  # Cubic root of 10 growth
            xa = x0 - step
            xb = x0 + step
            fa = call_f(xa)
            fb = call_f(xb)
            
            if fa is not None and is_good(fa):
                return None
            if fb is not None and is_good(fb):
                return None
            
            if fa is not None and fb is not None and fa * fb < 0:
                return (xa, xb, fa, fb)
        
        # Try asymmetric brackets
        for i in range(1, 15):
            if evals >= MAX_EVALS - 50 or time.monotonic() > DEADLINE:
                break
            step = base_step * (2 ** i)
            # Asymmetric: go further in one direction
            xa = x0 - step
            xb = x0 + step * 0.5
            fa = call_f(xa)
            fb = call_f(xb)
            if fa is not None and fb is not None and fa * fb < 0:
                return (xa, xb, fa, fb)
            
            xa = x0 + step
            xb = x0 - step * 0.5
            fa = call_f(xa)
            fb = call_f(xb)
            if fa is not None and fb is not None and fa * fb < 0:
                return (xa, xb, fa, fb)
        
        return None
    
    def brent_solve(xa, xb, fa, fb):
        """
        Use scipy's brentq with our evaluation counter.
        """
        nonlocal evals, best_x, best_abs
        
        # Wrap f to use our counter
        def wrapped_f(x):
            return call_f(x)
        
        try:
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
                return None
            
            # brentq with tight tolerances
            result = brentq(
                wrapped_f, xa, xb,
                xtol=1e-14,
                rtol=1e-14,
                maxiter=100
            )
            
            # Evaluate at the result to update best
            fr = call_f(result)
            if fr is not None and is_good(fr):
                return result
            
            return None
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, RuntimeError):
            return None
    
    def secant_fallback(x0, x1, max_iter=30):
        """
        Secant method as fallback when no bracket is found.
        """
        nonlocal evals, best_x, best_abs
        
        x_prev = float(x0)
        x_curr = float(x1)
        f_prev = call_f(x_prev)
        f_curr = call_f(x_curr)
        
        if f_prev is None or f_curr is None:
            return None
        if is_good(f_prev):
            return x_prev
        if is_good(f_curr):
            return x_curr
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
                break
            
            denom = f_curr - f_prev
            if denom == 0 or math.isnan(denom) or math.isinf(denom):
                break
            
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            
            # Damping for stability
            if abs(x_new - x_curr) > 10.0 * max(1.0, abs(x_curr)):
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
        
        return None
    
    def newton_fallback(x0, max_iter=30):
        """
        Newton's method with forward difference as last resort.
        """
        nonlocal evals, best_x, best_abs
        
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return None
        if is_good(fx):
            return x
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 5:
                break
            
            h = 1e-8 * max(1.0, abs(x))
            xp = x + h
            fpx = call_f(xp)
            if fpx is None:
                break
            
            d = (fpx - fx) / h
            if d == 0 or math.isnan(d) or math.isinf(d):
                break
            
            x_new = x - fx / d
            
            # Safeguard
            if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
            
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            
            if is_good(fx_new):
                return x_new
            
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                break
            
            # If diverging, stop
            if abs(fx_new) > abs(fx) * 2.0:
                break
            
            x = x_new
            fx = fx_new
        
        return None
    
    # Main strategy
    x0 = float(x0)
    
    # Step 1: Check if x0 is already a root
    fx0 = call_f(x0)
    if fx0 is not None and is_good(fx0):
        return x0
    
    # Step 2: Try to find a bracket and use Brent's method
    bracket = find_bracket(x0, max_expansions=25)
    if bracket is not None:
        xa, xb, fa, fb = bracket
        result = brent_solve(xa, xb, fa, fb)
        if result is not None:
            return result
    
    # Step 3: If no bracket, try secant with various starting points
    h = 1e-4 * max(1.0, abs(x0))
    
    # Secant from x0, x0+h
    result = secant_fallback(x0, x0 + h, max_iter=25)
    if result is not None:
        return result
    
    # Secant from x0-h, x0
    result = secant_fallback(x0 - h, x0, max_iter=25)
    if result is not None:
        return result
    
    # Secant from x0-h, x0+h
    result = secant_fallback(x0 - h, x0 + h, max_iter=25)
    if result is not None:
        return result
    
    # Step 4: Try Newton's method
    result = newton_fallback(x0, max_iter=20)
    if result is not None:
        return result
    
    # Step 5: Try different scales for bracket finding
    for scale in [1e-6, 1e-5, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 30:
            break
        
        base = scale * max(1.0, abs(x0))
        bracket = find_bracket(x0, max_expansions=10)
        if bracket is not None:
            xa, xb, fa, fb = bracket
            result = brent_solve(xa, xb, fa, fb)
            if result is not None:
                return result
        
        # Also try secant with this scale
        result = secant_fallback(x0 - base, x0 + base, max_iter=15)
        if result is not None:
            return result
    
    # Step 6: Random restarts if budget remains
    if evals < MAX_EVALS - 20 and time.monotonic() < DEADLINE - 5:
        rng = random.Random(123)
        for _ in range(8):
            if evals >= MAX_EVALS - 10 or time.monotonic() > DEADLINE - 3:
                break
            
            # Random perturbation
            scale = max(1.0, abs(best_x)) * rng.uniform(0.01, 0.5)
            x_try = best_x + rng.uniform(-scale, scale)
            
            # Try bracket finding from this point
            bracket = find_bracket(x_try, max_expansions=8)
            if bracket is not None:
                xa, xb, fa, fb = bracket
                result = brent_solve(xa, xb, fa, fb)
                if result is not None:
                    return result
            
            # Try secant
            h2 = 1e-4 * max(1.0, abs(x_try))
            result = secant_fallback(x_try, x_try + h2, max_iter=15)
            if result is not None:
                return result
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
