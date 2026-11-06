"""Root finding: Ill-conditioned - Robust hybrid solver with Brent's method."""
import time
import math

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: 
    1. Newton with robust derivative estimation.
    2. Brent's method for bracketing (faster than bisection).
    3. Secant method as a fallback.
    """
    start_time = time.time()
    MAX_WALL = 145.0  # Safety margin before 150s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    best_x = float(x0)
    best_fx = float('inf')
    
    def eval_f(x):
        nonlocal best_x, best_fx
        eval_count[0] += 1
        fx = f(x)
        if math.isfinite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        return fx
    
    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL
    
    def check_time():
        return (time.time() - start_time) < MAX_WALL

    def newton_step(x, fx, max_iter=20):
        """
        Newton's method with adaptive central difference derivative.
        Returns (x, fx, success).
        """
        for i in range(max_iter):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
            if abs(fx) <= TARGET_TOL:
                return x, fx, True
            
            # Adaptive step for derivative
            # Use central difference for better accuracy
            h = 1e-5 * max(1.0, abs(x))
            # Ensure h doesn't cause overflow or underflow
            if h == 0: h = 1e-5
            
            try:
                fx_plus = eval_f(x + h)
                fx_minus = eval_f(x - h)
                if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
                    break
                d = (fx_plus - fx_minus) / (2 * h)
            except Exception:
                break
            
            if d == 0 or not math.isfinite(d):
                break
            
            step = fx / d
            x_new = x - step
            
            # Safeguard: if step is too large, damp it
            # Limit step to 10x current magnitude or a fixed large bound
            max_step = 10 * max(1.0, abs(x))
            if abs(step) > max_step:
                x_new = x + math.copysign(max_step, step)
            
            fx_new = eval_f(x_new)
            if not math.isfinite(fx_new):
                break
            
            x, fx = x_new, fx_new
        
        return x, fx, is_good(x, fx)

    def brent(a, b, fa, fb, max_iter=100):
        """
        Brent's method for finding a root in [a, b] where fa*fb < 0.
        This is faster than bisection and robust.
        """
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return None, None, False
            
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            return None, None, False

        # Initialize variables
        c = a
        fc = fa
        d = b - a
        e = d
        
        for i in range(max_iter):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
                
            # Check convergence
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            if abs(b - a) <= 1e-12 * max(1.0, abs(b)):
                return b, fb, True
            
            # Inverse quadratic interpolation
            mflag = True
            if abs(e) > 1e-12:
                # Check if quadratic interpolation is possible
                if abs(d) > 1e-12 and abs(fb) > abs(fc):
                    # Secant step
                    s = a - fa * (b - a) / (fb - fa)
                    # Check if s is within reasonable bounds
                    if ((s - (2*a + b)/2) * (s - (a + 2*b)/2) < 0 and abs(s - b) < abs(b - c)):
                        mflag = False
                    else:
                        # Bisection
                        s = (a + b) / 2
                        e = d
                else:
                    # Bisection
                    s = (a + b) / 2
                    e = d
            
            if mflag:
                d = s - b
                e = d
            else:
                # Update previous step size
                e = d
            
            # Evaluate f(s)
            fs = eval_f(s)
            if not math.isfinite(fs):
                break
                
            if abs(fs) <= TARGET_TOL:
                return s, fs, True
                
            # Update bracket
            if fa * fs < 0:
                b, fb = s, fs
            else:
                a, fa = s, fs
                
            # Update c and fc
            if abs(fb) < abs(fc):
                c, fc = b, fb
            else:
                c, fc = a, fa
                
            d = b - c
            
        return b, fb, is_good(b, fb)

    # 1. Initial Evaluation
    x = float(x0)
    fx = eval_f(x)
    
    if is_good(x, fx):
        return x
    
    # 2. Try Newton from x0
    x_new, fx_new, success = newton_step(x, fx, max_iter=20)
    if success:
        return x_new
    
    # 3. Search for a bracket
    # Expand around x0
    h = 1e-3 * max(1.0, abs(x0))
    bracket_found = False
    a = b = fa = fb = None
    
    for _ in range(30):
        if not check_time() or eval_count[0] >= MAX_EVALS:
            break
            
        a = x0 - h
        b = x0 + h
        fa = eval_f(a)
        fb = eval_f(b)
        
        if not (math.isfinite(fa) and math.isfinite(fb)):
            h *= 2.0
            continue
            
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
            
        if fa * fb < 0:
            bracket_found = True
            break
            
        h *= 2.0

    # 4. If bracket found, use Brent's method
    if bracket_found:
        x_root, f_root, success = brent(a, b, fa, fb, max_iter=100)
        if success:
            return x_root
        # If Brent failed but got close, try Newton from the result
        if x_root is not None and math.isfinite(f_root):
            x_new, fx_new, success = newton_step(x_root, f_root, max_iter=10)
            if success:
                return x_new

    # 5. Fallback: Secant method from x0 and a nearby point
    if eval_count[0] < MAX_EVALS and check_time():
        h_sec = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h_sec
        f1 = eval_f(x1)
        
        if is_good(x1, f1):
            return x1
            
        # Secant iteration
        x_prev = x0
        x_curr = x1
        f_prev = fx
        f_curr = f1
        
        for _ in range(30):
            if not check_time() or eval_count[0] >= MAX_EVALS:
                break
            if abs(f_curr) <= TARGET_TOL:
                return x_curr
            if f_curr == f_prev:
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            if not math.isfinite(x_new):
                break
            f_new = eval_f(x_new)
            if not math.isfinite(f_new):
                break
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new

    # 6. Return best result found
    return best_x

# EVOLVE-BLOCK-END
