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
    MAX_WALL = 110.0  # Safety margin before 120s limit
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
    
    def is_good(x, fx=None):
        """Check if x is a good root."""
        if fx is None:
            fx = eval_f(x)
        return abs(fx) <= TARGET_TOL
    
    def check_time():
        """Check if we're close to the time limit."""
        return (time.time() - start_time) > MAX_WALL
    
    def newton_step(x, fx):
        """Take one Newton step with finite difference derivative."""
        h = 1e-6 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            return x, d
        x_new = x - fx / d
        if not math.isfinite(x_new):
            return x, d
        return x_new, d
    
    # First, try to get a good initial point with Newton
    x = float(x0)
    best_x = x
    best_fx = eval_f(x)
    best_abs_fx = abs(best_fx)
    
    if is_good(x, best_fx):
        return x
    
    # Try Newton's method first (fast if it works)
    for i in range(100):
        if check_time() or eval_count[0] >= MAX_EVALS:
            break
        
        fx = best_fx
        x_new, d = newton_step(x, fx)
        
        if x_new == x:
            break
        
        fx_new = eval_f(x_new)
        
        if math.isfinite(fx_new) and abs(fx_new) < best_abs_fx:
            best_x = x_new
            best_fx = fx_new
            best_abs_fx = abs(fx_new)
            x = x_new
            
            if is_good(x, fx_new):
                return x
        else:
            # Newton step didn't help, try a smaller step
            step_size = abs(x_new - x)
            for frac in [0.5, 0.25, 0.1]:
                if check_time() or eval_count[0] >= MAX_EVALS:
                    break
                x_try = x + frac * (x_new - x)
                fx_try = eval_f(x_try)
                if math.isfinite(fx_try) and abs(fx_try) < best_abs_fx:
                    best_x = x_try
                    best_fx = fx_try
                    best_abs_fx = abs(fx_try)
                    x = x_try
                    if is_good(x, fx_try):
                        return x
                    break
            else:
                # No improvement, break out of Newton
                break
    
    if is_good(best_x, best_fx):
        return best_x
    
    # Try secant method
    if not check_time() and eval_count[0] < MAX_EVALS:
        # Set up secant with two points
        x1 = best_x
        f1 = best_fx
        
        # Get a second point
        h = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + h
        f2 = eval_f(x2)
        
        if math.isfinite(f2) and abs(f2) < best_abs_fx:
            best_x = x2
            best_fx = f2
            best_abs_fx = abs(f2)
            if is_good(x2, f2):
                return x2
        
        for i in range(200):
            if check_time() or eval_count[0] >= MAX_EVALS:
                break
            
            if abs(f2 - f1) < 1e-15:
                break
            
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x3):
                break
            
            f3 = eval_f(x3)
            
            if math.isfinite(f3) and abs(f3) < best_abs_fx:
                best_x = x3
                best_fx = f3
                best_abs_fx = abs(f3)
                if is_good(x3, f3):
                    return x3
            
            x1, f1 = x2, f2
            x2, f2 = x3, f3
    
    if is_good(best_x, best_fx):
        return best_x
    
    # Try bisection-like approach: scan for sign changes
    if not check_time() and eval_count[0] < MAX_EVALS:
        # Try to find a bracket around x0
        scale = max(1.0, abs(float(x0)))
        f_x0 = eval_f(float(x0))
        
        for mult in [1e-2, 1e-1, 1, 10, 100]:
            if check_time() or eval_count[0] >= MAX_EVALS:
                break
            
            a = float(x0) - mult * scale
            b = float(x0) + mult * scale
            
            fa = eval_f(a)
            fb = eval_f(b)
            
            if not math.isfinite(fa) or not math.isfinite(fb):
                continue
            
            if abs(fa) < best_abs_fx:
                best_x = a
                best_fx = fa
                best_abs_fx = abs(fa)
                if is_good(a, fa):
                    return a
            
            if abs(fb) < best_abs_fx:
                best_x = b
                best_fx = fb
                best_abs_fx = abs(fb)
                if is_good(b, fb):
                    return b
            
            # Check for sign change
            if fa * fb < 0:
                # Bisection
                for j in range(100):
                    if check_time() or eval_count[0] >= MAX_EVALS:
                        break
                    
                    mid = (a + b) / 2
                    fm = eval_f(mid)
                    
                    if not math.isfinite(fm):
                        break
                    
                    if abs(fm) < best_abs_fx:
                        best_x = mid
                        best_fx = fm
                        best_abs_fx = abs(fm)
                        if is_good(mid, fm):
                            return mid
                    
                    if fm * fa < 0:
                        b = mid
                        fb = fm
                    else:
                        a = mid
                        fa = fm
                
                break
    
    # Final check
    if is_good(best_x, best_fx):
        return best_x
    
    return best_x

# EVOLVE-BLOCK-END
