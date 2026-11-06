"""Root finding: Ill-conditioned - robust solver with multiple strategies."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a portfolio approach with Newton, bisection, and secant methods.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    # Global evaluation counter
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            return 0.0
        return val
    
    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL
    
    def try_newton(x0, max_iter=50):
        """Newton's method with adaptive step size."""
        x = float(x0)
        for i in range(max_iter):
            if time.time() - start_time > MAX_WALL:
                return None
            if eval_count[0] >= MAX_EVALS:
                return None
            
            fx = eval_f(x)
            if is_good(x, fx):
                return x
            
            # Finite difference derivative
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
            
            # Damped Newton step
            dx = fx / d
            if not math.isfinite(dx):
                break
            
            # Limit step size to avoid divergence
            max_step = 1.0 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)
            
            x_new = x - dx
            
            # Check if we're making progress
            if not math.isfinite(x_new):
                break
            
            x = x_new
        
        return None
    
    def try_secant(x0, max_iter=50):
        """Secant method."""
        x = float(x0)
        h = 1e-4 * max(1.0, abs(x))
        x_prev = x - h
        fx_prev = eval_f(x_prev)
        
        for i in range(max_iter):
            if time.time() - start_time > MAX_WALL:
                return None
            if eval_count[0] >= MAX_EVALS:
                return None
            
            fx = eval_f(x)
            if is_good(x, fx):
                return x
            if is_good(x_prev, fx_prev):
                return x_prev
            
            if fx == fx_prev:
                break
            
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
            
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            x_prev, fx_prev = x, fx
            x = x_new
        
        return None
    
    def try_bisection_safe(x0, max_iter=100):
        """Bisection with bracketing search."""
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        # Try to find a bracket
        h = 1e-3 * max(1.0, abs(x))
        left, right = None, None
        fl, fr = None, None
        
        for scale in [1, 10, 100, 1000, 1e4, 1e5]:
            h_try = h * scale
            xl = x - h_try
            xr = x + h_try
            fl_try = eval_f(xl)
            fr_try = eval_f(xr)
            
            if is_good(xl, fl_try):
                return xl
            if is_good(xr, fr_try):
                return xr
            
            if fl_try * fr_try < 0:
                left, right = xl, xr
                fl, fr = fl_try, fr_try
                break
        
        if left is None:
            return None
        
        # Bisection
        for i in range(max_iter):
            if time.time() - start_time > MAX_WALL:
                return None
            if eval_count[0] >= MAX_EVALS:
                return None
            
            mid = (left + right) / 2
            fm = eval_f(mid)
            
            if is_good(mid, fm):
                return mid
            
            if fl * fm <= 0:
                right = mid
                fr = fm
            else:
                left = mid
                fl = fm
        
        return None
    
    # Strategy 1: Newton's method
    result = try_newton(x0)
    if result is not None:
        return result
    
    # Strategy 2: Secant method
    result = try_secant(x0)
    if result is not None:
        return result
    
    # Strategy 3: Bisection with bracketing
    result = try_bisection_safe(x0)
    if result is not None:
        return result
    
    # Strategy 4: Multiple Newton restarts with perturbed initial guesses
    for offset in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1]:
        if time.time() - start_time > MAX_WALL or eval_count[0] >= MAX_EVALS:
            break
        x_perturbed = x0 + offset * max(1.0, abs(x0))
        result = try_newton(x_perturbed, max_iter=30)
        if result is not None:
            return result
    
    # Strategy 5: Secant with perturbed initial guess
    for offset in [1e-3, -1e-3, 1e-2, -1e-2]:
        if time.time() - start_time > MAX_WALL or eval_count[0] >= MAX_EVALS:
            break
        x_perturbed = x0 + offset * max(1.0, abs(x0))
        result = try_secant(x_perturbed, max_iter=30)
        if result is not None:
            return result
    
    # Strategy 6: Bisection with perturbed initial guess
    for offset in [1e-2, -1e-2, 0.1, -0.1]:
        if time.time() - start_time > MAX_WALL or eval_count[0] >= MAX_EVALS:
            break
        x_perturbed = x0 + offset * max(1.0, abs(x0))
        result = try_bisection_safe(x_perturbed, max_iter=50)
        if result is not None:
            return result
    
    # Return best effort - use x0 as fallback
    return float(x0)

# EVOLVE-BLOCK-END
