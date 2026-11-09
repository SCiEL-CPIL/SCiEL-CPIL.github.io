"""Root finding for ill-conditioned equations: robust adaptive hybrid."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s limit
    max_evals = 500
    eval_count = [0]
    
    def safe_eval(x):
        if eval_count[0] >= max_evals:
            raise RuntimeError("Eval budget exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time deadline exceeded")
        eval_count[0] += 1
        return f(x)
    
    def check_time():
        if time.time() > deadline:
            raise RuntimeError("Time deadline exceeded")
    
    x = float(x0)
    best_x = x
    best_f = safe_eval(x)
    best_abs = abs(best_f)
    
    # Phase 1: Safeguarded Newton with bracketing
    a = None
    b = None
    fa = None
    fb = None
    
    for i in range(80):
        check_time()
        
        # Update best
        if best_abs <= 1e-8:
            return best_x
        
        # Compute derivative using central difference if possible
        h = 1e-6 * max(1.0, abs(x))
        try:
            if eval_count[0] < max_evals - 2:
                fp = safe_eval(x + h)
                fm = safe_eval(x - h)
                d = (fp - fm) / (2 * h)
            else:
                fp = safe_eval(x + h)
                d = (fp - best_f) / h
        except RuntimeError:
            break
        
        if d == 0 or not np.isfinite(d):
            # Try secant step if we have a previous point
            break
        
        # Newton step
        step = best_f / d
        x_new = x - step
        
        # Clip step to avoid runaway
        if abs(step) > 10 * max(1.0, abs(x)):
            step = math.copysign(10 * max(1.0, abs(x)), step)
            x_new = x - step
        
        # Check if we can bracket
        if a is not None and b is not None:
            # Ensure x_new is within bracket or update bracket
            if (a <= x_new <= b) or (b <= x_new <= a):
                pass
            else:
                # Try to maintain bracket
                pass
        
        fx_new = safe_eval(x_new)
        abs_new = abs(fx_new)
        
        if abs_new < best_abs:
            best_x = x_new
            best_f = fx_new
            best_abs = abs_new
            x = x_new
            
            # Update bracket
            if a is None or fa is None:
                a = x
                fa = best_f
            if (fa > 0 and best_f < 0) or (fa < 0 and best_f > 0):
                b = x
                fb = best_f
        else:
            # Newton step didn't improve; try bisection or secant
            if a is not None and b is not None and fa is not None and fb is not None:
                # Use bisection
                if (fa > 0) == (fb > 0):
                    # Bracket is invalid, reset
                    a = None
                    b = None
                else:
                    mid = (a + b) / 2.0
                    fm = safe_eval(mid)
                    if abs(fm) < best_abs:
                        best_x = mid
                        best_f = fm
                        best_abs = abs(fm)
                    if (fa > 0) == (fm > 0):
                        a = mid
                        fa = fm
                    else:
                        b = mid
                        fb = fm
                    x = mid
            else:
                # Try a damped Newton or secant
                x = x - 0.5 * step
                if not np.isfinite(x):
                    break
    
    # Phase 2: If we have a bracket, use Brent's method
    if a is not None and b is not None and fa is not None and fb is not None:
        if (fa > 0) != (fb > 0):
            try:
                if time.time() < deadline and eval_count[0] < max_evals - 50:
                    x_brent = brentq(safe_eval, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                    fx_brent = safe_eval(x_brent)
                    if abs(fx_brent) < best_abs:
                        best_x = x_brent
                        best_f = fx_brent
                        best_abs = abs(fx_brent)
            except Exception:
                pass
    
    # Phase 3: Final polish with tight tolerance
    if best_abs > 1e-8 and best_abs < 1e-4:
        # Try a few more Newton steps with tighter tolerance
        x = best_x
        fx = best_f
        for j in range(20):
            check_time()
            if abs(fx) <= 1e-12:
                return x
            h = 1e-8 * max(1.0, abs(x))
            try:
                fp = safe_eval(x + h)
                d = (fp - fx) / h
                if d == 0 or not np.isfinite(d):
                    break
                x_new = x - fx / d
                fx_new = safe_eval(x_new)
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
                    if abs(fx) < best_abs:
                        best_x = x
                        best_f = fx
                        best_abs = abs(fx)
                else:
                    break
            except RuntimeError:
                break
    
    return best_x
# EVOLVE-BLOCK-END
