import time
import math
import numpy as np
from scipy.optimize import minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    deadline = 110.0  # Leave margin for 120s limit
    call_count = 0
    max_calls = 480
    
    best_x = float(x0)
    best_f2 = float('inf')
    best_abs_f = float('inf')
    
    def obj(x):
        nonlocal call_count, best_x, best_f2, best_abs_f
        if time.time() - start_time > deadline:
            raise TimeoutError("Time limit exceeded")
        call_count += 1
        if call_count > max_calls:
            raise TimeoutError("Call limit exceeded")
        fx = f(x)
        if not math.isfinite(fx):
            return float('inf')
        f2 = fx * fx
        abs_fx = abs(fx)
        if abs_fx < best_abs_f:
            best_abs_f = abs_fx
            best_f2 = f2
            best_x = x
        return f2

    def illinois(fa, a, fb, b):
        """Illinois method: bisection + linear interpolation."""
        nonlocal call_count
        for _ in range(100):
            if call_count >= max_calls or time.time() - start_time > deadline:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            # Illinois: linear interpolation, but if it's outside [a,b], use midpoint
            if fa == fb:
                mid = (a + b) / 2.0
            else:
                mid = (a * fb - b * fa) / (fb - fa)
            if not (a < mid < b) or not math.isfinite(mid):
                mid = (a + b) / 2.0
            
            if call_count >= max_calls or time.time() - start_time > deadline:
                break
                
            call_count += 1
            fx = f(mid)
            if not math.isfinite(fx):
                break
            fm = fx
            abs_fm = abs(fm)
            if abs_fm < best_abs_f:
                best_abs_f = abs_fm
                best_f2 = fm * fm
                best_x = mid
                
            if abs_fm <= 1e-8:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                # Illinois trick: shrink the side with smaller |f|
                if abs(fa) < abs(fb):
                    fa = fa / 2.0
                else:
                    fb = fb / 2.0
                a = mid
                fa = fm
        # Return the better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    # Try expanding intervals
    # Start with a small interval, expand if no root found
    widths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
    
    # First, do a quick local scan to find a bracket for Illinois
    # This is faster than minimize_scalar for finding the bracket
    x = float(x0)
    fx = None
    if call_count < max_calls and time.time() - start_time < deadline:
        call_count += 1
        fx = f(x)
        if math.isfinite(fx):
            best_abs_f = abs(fx)
            best_f2 = fx * fx
            best_x = x
            if abs(fx) <= 1e-8:
                return x
        else:
            fx = float('nan')
    
    if fx is not None and math.isfinite(fx):
        # Try to find a bracket locally with a few steps
        step = 0.1 * max(1.0, abs(x))
        # Try positive direction
        for i in range(1, 20):
            if call_count >= max_calls or time.time() - start_time > deadline:
                break
            x_pos = x + i * step
            call_count += 1
            f_pos = f(x_pos)
            if not math.isfinite(f_pos):
                break
            abs_f_pos = abs(f_pos)
            if abs_f_pos < best_abs_f:
                best_abs_f = abs_f_pos
                best_f2 = f_pos * f_pos
                best_x = x_pos
            if abs_f_pos <= 1e-8:
                return x_pos
            if fx * f_pos < 0:
                # Found bracket [x, x_pos]
                result = illinois(fx, x, f_pos, x_pos)
                if result is not None and best_abs_f <= 1e-8:
                    return best_x
                # If Illinois didn't converge to 1e-8, fall back to minimize_scalar on this bracket
                # But first, let's see if best_x is good enough
                if best_abs_f <= 1e-8:
                    return best_x
                # Try minimize_scalar on the bracket found
                try:
                    res = minimize_scalar(obj, bounds=(x, x_pos), method='bounded', 
                                          options={'maxiter': 30, 'xatol': 1e-12, 'fatol': 1e-16})
                    if np.sqrt(res.fun) <= 1e-8:
                        return res.x
                except Exception:
                    pass
                return best_x
            step *= 1.2  # Expand adaptively, but not too fast to avoid missing roots

        # Try negative direction
        step = 0.1 * max(1.0, abs(x))
        for i in range(1, 20):
            if call_count >= max_calls or time.time() - start_time > deadline:
                break
            x_neg = x - i * step
            call_count += 1
            f_neg = f(x_neg)
            if not math.isfinite(f_neg):
                break
            abs_f_neg = abs(f_neg)
            if abs_f_neg < best_abs_f:
                best_abs_f = abs_f_neg
                best_f2 = f_neg * f_neg
                best_x = x_neg
            if abs_f_neg <= 1e-8:
                return x_neg
            if fx * f_neg < 0:
                # Found bracket [x_neg, x]
                result = illinois(f_neg, x_neg, fx, x)
                if best_abs_f <= 1e-8:
                    return best_x
                # Try minimize_scalar on the bracket found
                try:
                    res = minimize_scalar(obj, bounds=(x_neg, x), method='bounded', 
                                          options={'maxiter': 30, 'xatol': 1e-12, 'fatol': 1e-16})
                    if np.sqrt(res.fun) <= 1e-8:
                        return res.x
                except Exception:
                    pass
                return best_x
            step *= 1.2

    # If no bracket found quickly, use the minimize_scalar approach with expanding intervals
    for w in widths:
        if call_count >= max_calls or time.time() - start_time > deadline:
            break
        lo = x0 - w
        hi = x0 + w
        try:
            # maxiter=50 is usually enough for bounded Brent
            # xatol=1e-12 ensures high precision in x
            res = minimize_scalar(obj, bounds=(lo, hi), method='bounded', 
                                  options={'maxiter': 40, 'xatol': 1e-12, 'fatol': 1e-16})
            x_opt = res.x
            # Check the actual residual
            # Note: res.fun is f(x_opt)^2
            if np.sqrt(res.fun) <= 1e-8:
                return x_opt
        except TimeoutError:
            break
        except Exception:
            continue
            
    # If no root found with |f(x)| <= 1e-8, return the best x found
    # This might still be a failure if |f(best_x)| > 1e-8, but it's the best we have
    return best_x
