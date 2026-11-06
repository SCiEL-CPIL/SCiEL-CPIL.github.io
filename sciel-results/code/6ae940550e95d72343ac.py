"""Root finding: Ill-conditioned - Robust hybrid solver with Brent's method."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: Brent's method, Newton with safeguarding,
    bisection fallback, and multiple restarts if needed.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    cache = {}
    
    def eval_f(x):
        """Evaluate f with caching."""
        # Use a rounded key for caching to handle floating point issues
        key = round(x, 15)
        if key in cache:
            return cache[key]
        
        eval_count[0] += 1
        val = f(x)
        cache[key] = val
        return val
    
    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL
    
    def check_time():
        return (time.time() - start_time) < MAX_WALL
    
    def check_budget():
        return eval_count[0] < MAX_EVALS and check_time()
    
    def newton_step(x, fx, max_iter=30):
        """Newton's method with finite difference derivative."""
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(fx) <= TARGET_TOL:
                return x, fx, True
            
            # Adaptive step size for derivative
            h = 1e-8 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or math.isnan(d) or math.isinf(d):
                break
            
            # Damped Newton step
            x_new = x - fx / d
            
            # Safeguard: keep step within reasonable bounds
            step = x_new - x
            if abs(step) > 5 * max(1.0, abs(x)):
                x_new = x + math.copysign(5 * max(1.0, abs(x)), step)
            
            fx_new = eval_f(x_new)
            if math.isnan(fx_new) or math.isinf(fx_new):
                break
            
            x, fx = x_new, fx_new
        
        return x, fx, is_good(x, fx)
    
    def bisection(a, b, fa, fb, max_iter=60):
        """Bisection method."""
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(fa) <= TARGET_TOL:
                return a, fa, True
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            
            if math.isnan(fm) or math.isinf(fm):
                break
            
            if abs(fm) <= TARGET_TOL:
                return mid, fm, True
            
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        
        # Return the point with smaller |f|
        if abs(fa) <= abs(fb):
            return a, fa, is_good(a, fa)
        else:
            return b, fb, is_good(b, fb)
    
    def secant(x0, x1, f0, f1, max_iter=30):
        """Secant method."""
        x, f_prev, f_curr = x0, f0, f1
        for i in range(max_iter):
            if not check_budget():
                break
            if abs(f_curr) <= TARGET_TOL:
                return x, f_curr, True
            
            if f_curr == f_prev:
                break
            
            x_new = x - f_curr * (x - x1) / (f_curr - f_prev)
            
            # Safeguard
            step = x_new - x
            if abs(step) > 5 * max(1.0, abs(x)):
                x_new = x + math.copysign(5 * max(1.0, abs(x)), step)
            
            f_new = eval_f(x_new)
            if math.isnan(f_new) or math.isinf(f_new):
                break
            
            x1, x = x, x_new
            f_prev, f_curr = f_curr, f_new
        
        return x, f_curr, is_good(x, f_curr)
    
    def brent(a, b, fa, fb, max_iter=100):
        """
        Brent's method: combines bisection, secant, and inverse quadratic interpolation.
        """
        c = a
        fc = fa
        d = b - a
        e = d
        
        for i in range(max_iter):
            if not check_budget():
                break
            
            # Check if we've converged
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            if abs(fc) <= TARGET_TOL:
                return c, fc, True
            
            # Check for sign change
            if fa * fb >= 0:
                # No sign change, use the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a, fa, is_good(a, fa)
                else:
                    return b, fb, is_good(b, fb)
            
            # Inverse quadratic interpolation
            m1 = (fb - fc) / (b - c)
            m2 = (fb - fa) / (b - a)
            
            if abs(m1 - m2) > 1e-15:
                s = b - fb * (fb - fc) * m2 / (m1 * m2)
            else:
                s = b - fb * (b - a) / (2 * (fb - fa))
            
            # Conditions for using the interpolation step
            cond1 = (1.5 * m2 * (a - b) + fb) * (a - b) > 0
            cond2 = s < min(a, b) + 0.5 * (b - a)
            cond3 = s > max(a, b) - 0.5 * (b - a)
            cond4 = abs(e) < 1e-15 or abs(s - b) >= 0.5 * abs(e)
            
            if not (cond1 and cond2 and cond3 and cond4):
                # Use bisection
                s = (a + b) / 2.0
                e = d
            
            d = b - s
            fs = eval_f(s)
            
            if math.isnan(fs) or math.isinf(fs):
                break
            
            if abs(fs) <= TARGET_TOL:
                return s, fs, True
            
            if fa * fs < 0:
                b, fb = s, fs
            else:
                a, fa = s, fs
            
            c, fc = b, fb
            e, d = d, b - s
        
        # Return the best point
        if abs(fa) <= abs(fb):
            return a, fa, is_good(a, fa)
        else:
            return b, fb, is_good(b, fb)
    
    # Try initial point
    x = float(x0)
    fx = eval_f(x)
    
    if is_good(x, fx):
        return x
    
    # Strategy 1: Newton from x0 (fast if well-conditioned)
    x, fx, success = newton_step(x, fx, max_iter=20)
    if success:
        return x
    
    # Strategy 2: Try to find a bracket for Brent's method
    # Expand around x0
    h = 1e-3 * max(1.0, abs(x0))
    found_bracket = False
    best_x, best_fx = x, fx
    
    for expansion in range(25):
        if not check_budget():
            break
        
        a = x0 - h
        b = x0 + h
        fa = eval_f(a)
        fb = eval_f(b)
        
        if abs(fa) <= TARGET_TOL:
            return a
        if abs(fb) <= TARGET_TOL:
            return b
        
        if fa * fb < 0:
            # Found bracket, use Brent's method
            x, fx, success = brent(a, b, fa, fb, max_iter=80)
            if success:
                return x
            
            # If Brent didn't fully converge, try Newton from the best point
            if abs(fx) < abs(best_fx):
                best_x, best_fx = x, fx
            
            found_bracket = True
            break
        
        # Update best
        if abs(fa) < abs(best_fx):
            best_x, best_fx = a, fa
        if abs(fb) < abs(best_fx):
            best_x, best_fx = b, fb
        
        h *= 2.0
    
    # Strategy 3: Secant from nearby points
    if check_budget():
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        f1 = eval_f(x1)
        
        if abs(f1) <= TARGET_TOL:
            return x1
        
        x, fx, success = secant(x0, x1, fx, f1, max_iter=30)
        if success:
            return x
    
    # Strategy 4: Try Newton from best point found
    if check_budget():
        x, fx, success = newton_step(best_x, best_fx, max_iter=25)
        if success:
            return x
    
    # Strategy 5: More aggressive search with multiple restarts
    if check_budget():
        offsets = [1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 5.0, -5.0]
        for offset in offsets:
            if not check_budget():
                break
            
            x_try = x0 + offset * max(1.0, abs(x0))
            fx_try = eval_f(x_try)
            
            if abs(fx_try) <= TARGET_TOL:
                return x_try
            
            if abs(fx_try) < abs(best_fx):
                best_x, best_fx = x_try, fx_try
            
            # Try Newton from this point
            x, fx, success = newton_step(x_try, fx_try, max_iter=15)
            if success:
                return x
    
    # Strategy 6: Try Brent with a wider bracket if not found before
    if not found_bracket and check_budget():
        # Try a few wider brackets
        for scale in [0.1, 0.5, 2.0, 10.0]:
            if not check_budget():
                break
            
            h = scale * max(1.0, abs(x0))
            a = x0 - h
            b = x0 + h
            fa = eval_f(a)
            fb = eval_f(b)
            
            if abs(fa) <= TARGET_TOL:
                return a
            if abs(fb) <= TARGET_TOL:
                return b
            
            if fa * fb < 0:
                x, fx, success = brent(a, b, fa, fb, max_iter=50)
                if success:
                    return x
                if abs(fx) < abs(best_fx):
                    best_x, best_fx = x, fx
                break
    
    # Return best result found
    return best_x

# EVOLVE-BLOCK-END
