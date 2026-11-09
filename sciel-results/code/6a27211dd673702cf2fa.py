"""Root finding for ill-conditioned equations: robust adaptive hybrid."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin before 150s limit
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
    
    # Phase 1: Safeguarded Newton with robust bracketing
    a = None
    b = None
    fa = None
    fb = None
    prev_x = None
    prev_f = None
    
    for i in range(100):
        check_time()
        
        if best_abs <= 1e-8:
            return best_x
        
        # Compute derivative using central difference
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
            # Fall back to secant if we have a previous point
            if prev_x is not None and prev_f is not None and abs(prev_x - x) > 1e-15:
                d_sec = (best_f - prev_f) / (x - prev_x)
                if d_sec != 0 and np.isfinite(d_sec):
                    step = best_f / d_sec
                    x_new = x - step
                    if abs(step) > 10 * max(1.0, abs(x)):
                        step = math.copysign(10 * max(1.0, abs(x)), step)
                        x_new = x - step
                else:
                    break
            else:
                break
        else:
            step = best_f / d
            x_new = x - step
        
        # Clip step to avoid runaway
        if abs(step) > 10 * max(1.0, abs(x)):
            step = math.copysign(10 * max(1.0, abs(x)), step)
            x_new = x - step
        
        if not np.isfinite(x_new):
            break
        
        fx_new = safe_eval(x_new)
        abs_new = abs(fx_new)
        
        # Update bracket
        if a is None:
            a = x
            fa = best_f
        if (fa > 0) != (fx_new > 0):
            b = x_new
            fb = fx_new
        elif abs_new < abs(fa):
            a = x_new
            fa = fx_new
        elif b is not None and (fb > 0) != (fx_new > 0):
            b = x_new
            fb = fx_new
        
        if abs_new < best_abs:
            prev_x = x
            prev_f = best_f
            best_x = x_new
            best_f = fx_new
            best_abs = abs_new
            x = x_new
        else:
            # Newton step didn't improve; use bisection if we have a valid bracket
            if a is not None and b is not None and fa is not None and fb is not None:
                if (fa > 0) != (fb > 0):
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
                    prev_x = x
                    prev_f = best_f
                else:
                    # Invalid bracket, reset
                    a = None
                    b = None
                    # Try damped Newton
                    x = x - 0.5 * step
                    if not np.isfinite(x):
                        break
            else:
                # No bracket; try damped Newton
                x = x - 0.5 * step
                if not np.isfinite(x):
                    break
    
    # Phase 2: If we have a bracket, use Brent's method
    if a is not None and b is not None and fa is not None and fb is not None:
        if (fa > 0) != (fb > 0):
            try:
                if time.time() < deadline and eval_count[0] < max_evals - 30:
                    x_brent = brentq(safe_eval, a, b, xtol=1e-14, rtol=1e-14, maxiter=200)
                    fx_brent = safe_eval(x_brent)
                    if abs(fx_brent) < best_abs:
                        best_x = x_brent
                        best_f = fx_brent
                        best_abs = abs(fx_brent)
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_abs > 1e-8:
        x = best_x
        fx = best_f
        for j in range(30):
            check_time()
            if abs(fx) <= 1e-14:
                return x
            h = 1e-10 * max(1.0, abs(x))
            try:
                if eval_count[0] < max_evals - 2:
                    fp = safe_eval(x + h)
                    fm = safe_eval(x - h)
                    d = (fp - fm) / (2 * h)
                else:
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
                    # Try secant-like step
                    if prev_x is not None and abs(prev_x - x) > 1e-15:
                        d_sec = (fx - prev_f) / (x - prev_x)
                        if d_sec != 0 and np.isfinite(d_sec):
                            x_new2 = x - fx / d_sec
                            fx_new2 = safe_eval(x_new2)
                            if abs(fx_new2) < abs(fx):
                                x = x_new2
                                fx = fx_new2
                                if abs(fx) < best_abs:
                                    best_x = x
                                    best_f = fx
                                    best_abs = abs(fx)
                            prev_x = x
                            prev_f = fx
                    break
            except RuntimeError:
                break
    
    return best_x
# EVOLVE-BLOCK-END
