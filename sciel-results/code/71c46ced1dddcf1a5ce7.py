"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Strategy:
    1. Try to find a sign-change bracket via adaptive steps.
    2. Use a safeguarded secant/bisection hybrid (SIQI-like) on the bracket.
    3. If no bracket is found, try Newton/secant with damping from x0.
    4. Deadline guard to stay within time limits.
    """
    deadline = time.time() + 145  # Leave 5s margin before 150s kill
    max_evals = 490
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            return val
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
    
    def time_check():
        return time.time() > deadline
    
    # First, evaluate at x0
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket
    a, b, fa, fb = None, None, None, None
    
    # Adaptive bracketing
    step = 1e-6 * max(1.0, abs(x))
    for i in range(30):
        if time_check():
            break
        # Try positive direction
        xp = x + step
        fxp = eval_f(xp)
        if fxp is None:
            break
        if fxp == 0:
            return xp
        if fa is None:
            a, fa = x, fx
        if fa * fxp < 0:
            a, fa = x, fx
            b, fb = xp, fxp
            break
        # Try negative direction
        xn = x - step
        fxn = eval_f(xn)
        if fxn is None:
            break
        if fxn == 0:
            return xn
        if fa is None:
            a, fa = x, fx
        if fa * fxn < 0:
            a, fa = x, fx
            b, fb = xn, fxn
            break
        
        # Expand step
        step *= 2.0
        if step > 1e10:
            break
    
    # If we have a bracket, use safeguarded secant/bisection
    if a is not None and b is not None:
        # Safeguarded secant method
        best_x, best_fx = x, fx
        for _ in range(200):
            if time_check():
                break
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            
            # Check convergence
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            
            # Secant step
            if fa != fb:
                x_sec = b - fb * (b - a) / (fb - fa)
            else:
                x_sec = (a + b) / 2
            
            # If secant step is outside bracket or too close to edge, use bisection
            if x_sec <= a or x_sec >= b or abs(x_sec - b) < 0.1 * abs(b - a):
                x_new = (a + b) / 2
            else:
                x_new = x_sec
            
            f_new = eval_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= 1e-8:
                return x_new
            
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
            
            if abs(f_new) < abs(best_fx):
                best_x, best_fx = x_new, f_new
            
            # Check if bracket is too small
            if abs(b - a) < 1e-15 * max(1.0, abs(a)):
                break
        
        # Return the best point found
        if abs(fa) < abs(fb):
            return a
        return b
    
    # No bracket found, try Newton's method with damping
    x = float(x0)
    best_x, best_fx = x, fx
    
    for _ in range(100):
        if time_check():
            break
        if abs(best_fx) <= 1e-8:
            return best_x
        
        # Finite difference derivative
        h = 1e-8 * max(1.0, abs(x))
        fp = eval_f(x + h)
        fm = eval_f(x - h)
        if fp is None or fm is None:
            break
        d = (fp - fm) / (2 * h)
        if d == 0 or not math.isfinite(d):
            break
        
        # Newton step
        x_new = x - fx / d
        
        # Damping: if step is too large, reduce it
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)
        
        f_new = eval_f(x_new)
        if f_new is None:
            break
        
        if abs(f_new) < abs(best_fx):
            best_x, best_fx = x_new, f_new
            x, fx = x_new, f_new
        else:
            # Step was bad, try half step
            x_half = x - 0.5 * (x_new - x)
            f_half = eval_f(x_half)
            if f_half is not None and abs(f_half) < abs(best_fx):
                best_x, best_fx = x_half, f_half
                x, fx = x_half, f_half
            else:
                # Give up on Newton
                break
        
        if abs(fx) <= 1e-8:
            return x
    
    return best_x
# EVOLVE-BLOCK-END
