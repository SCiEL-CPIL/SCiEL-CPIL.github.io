"""Root finder for 3 fixed oscillatory equations using minimize_scalar on (f(x))^2 with expanding bounds."""
import math
import time
from scipy.optimize import minimize_scalar


# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 118.0  # Safety margin for 120s limit
    MAX_CALLS = 495     # Stay under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _time_exceeded():
        return time.time() - start_time > TIME_LIMIT

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if _time_exceeded():
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if not isinstance(val, (int, float)):
            return float('nan')
        val = float(val)
        if not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = float(x)
        return val

    def _objective(x):
        """Squared residual to minimize."""
        fx = _f(x)
        if math.isnan(fx):
            return float('inf')
        return fx * fx

    x = float(x0)
    fx = _f(x)
    if abs(fx) <= 1e-8:
        return x

    # Try minimize_scalar with expanding bounds
    # Start with a moderate range and expand if needed
    radii = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]
    
    for radius in radii:
        if _time_exceeded() or calls >= MAX_CALLS - 10:
            break
        
        lower = x - radius
        upper = x + radius
        
        try:
            result = minimize_scalar(
                _objective,
                bounds=(lower, upper),
                method='bounded',
                options={'xatol': 1e-10, 'maxiter': 200}
            )
            
            if result.success and math.isfinite(result.x):
                f_result = _f(result.x)
                if not math.isnan(f_result):
                    if abs(f_result) <= 1e-8:
                        return float(result.x)
                    if abs(f_result) < best_abs_f:
                        best_x = float(result.x)
                        best_abs_f = abs(f_result)
                # If we found a very good result, stop expanding
                if best_abs_f < 1e-6:
                    break
            elif result.x is not None and math.isfinite(result.x):
                f_result = _f(result.x)
                if not math.isnan(f_result):
                    if abs(f_result) <= 1e-8:
                        return float(result.x)
                    if abs(f_result) < best_abs_f:
                        best_x = float(result.x)
                        best_abs_f = abs(f_result)
                if best_abs_f < 1e-6:
                    break
        except Exception:
            continue
    
    # If still not good enough, do a quick local bracketing search
    if best_abs_f > 1e-8 and not _time_exceeded() and calls < MAX_CALLS - 20:
        # Fine local scan for sign change
        local_radius = 2.0
        n_local = 40
        lower = x - local_radius
        upper = x + local_radius
        
        prev_x = None
        prev_f = None
        
        for i in range(n_local + 1):
            if _time_exceeded() or calls >= MAX_CALLS - 10:
                break
            
            cur_x = lower + i * (upper - lower) / n_local
            cur_f = _f(cur_x)
            
            if math.isnan(cur_f):
                continue
            
            if abs(cur_f) <= 1e-8:
                return cur_x
            
            if prev_x is not None and prev_f is not None and not math.isnan(prev_f):
                if prev_f * cur_f < 0:
                    # Found a bracket, use bisection
                    a, b = prev_x, cur_x
                    fa, fb = prev_f, cur_f
                    for _ in range(60):
                        if _time_exceeded() or calls >= MAX_CALLS - 5:
                            break
                        c = (a + b) / 2.0
                        fc = _f(c)
                        if math.isnan(fc):
                            break
                        if abs(fc) <= 1e-8:
                            return c
                        if fa * fc < 0:
                            b = c
                            fb = fc
                        else:
                            a = c
                            fa = fc
                    # Return the best of a or b
                    fa_final = _f(a) if not math.isnan(_f(a)) else fa
                    fb_final = _f(b) if not math.isnan(_f(b)) else fb
                    if abs(fa_final) < abs(fb_final):
                        if abs(fa_final) < best_abs_f:
                            best_x = a
                            best_abs_f = abs(fa_final)
                    else:
                        if abs(fb_final) < best_abs_f:
                            best_x = b
                            best_abs_f = abs(fb_final)
                    break
            
            prev_x = cur_x
            prev_f = cur_f
    
    return best_x

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
