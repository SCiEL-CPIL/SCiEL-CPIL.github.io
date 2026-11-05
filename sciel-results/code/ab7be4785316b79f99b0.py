"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if not math.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
            best_f = val
        return val
    
    def is_good(x, fx):
        if fx is None:
            return False
        return abs(fx) <= TOL
    
    def newton_raphson(x0, max_iter=60):
        nonlocal evals
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return x, None
        if is_good(x, fx):
            return x, fx
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            
            # Adaptive derivative step: try multiple scales
            h_base = 1e-5 * max(1.0, abs(x))
            d = None
            for scale in [1.0, 10.0, 0.1, 100.0, 0.01]:
                h = h_base * scale
                if h < 1e-12:
                    continue
                fx2 = call_f(x + h)
                if fx2 is None:
                    continue
                d_candidate = (fx2 - fx) / h
                if math.isfinite(d_candidate) and abs(d_candidate) > 1e-15:
                    d = d_candidate
                    break
            
            if d is None:
                # Try central difference
                h = 1e-5 * max(1.0, abs(x))
                fx_plus = call_f(x + h)
                fx_minus = call_f(x - h)
                if fx_plus is not None and fx_minus is not None:
                    d = (fx_plus - fx_minus) / (2 * h)
                if d is None or not math.isfinite(d) or abs(d) < 1e-15:
                    break
            
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            # Damping if step is too large
            max_step = 10 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + math.copysign(max_step, x_new - x)
            
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            
            if is_good(x_new, fx_new):
                return x_new, fx_new
            
            # Accept step if it reduces |f|, otherwise try smaller steps
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Try progressively smaller steps
                found = False
                for factor in [0.5, 0.25, 0.1, 0.05]:
                    x_trial = x - factor * (fx / d)
                    fx_trial = call_f(x_trial)
                    if fx_trial is not None and abs(fx_trial) < abs(fx):
                        x = x_trial
                        fx = fx_trial
                        found = True
                        break
                if not found:
                    break
        
        return x, fx
    
    def secant(x0, x1, max_iter=60):
        nonlocal evals
        x = float(x0)
        x_prev = float(x1)
        fx = call_f(x)
        fx_prev = call_f(x_prev)
        
        if fx is None or fx_prev is None:
            return x, fx if fx is not None else fx_prev
        if is_good(x, fx):
            return x, fx
        if is_good(x_prev, fx_prev):
            return x_prev, fx_prev
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            if fx == fx_prev:
                break
            
            denom = fx - fx_prev
            if abs(denom) < 1e-30:
                break
                
            x_new = x - fx * (x - x_prev) / denom
            if not math.isfinite(x_new):
                break
            
            # Damping for secant
            max_step = 10 * max(1.0, abs(x))
            if abs(x_new - x) > max_step:
                x_new = x + math.copysign(max_step, x_new - x)
            
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            
            if is_good(x_new, fx_new):
                return x_new, fx_new
            
            # Only accept if we're making progress
            if abs(fx_new) < abs(fx) or abs(fx_new) < abs(fx_prev):
                x_prev = x
                fx_prev = fx
                x = x_new
                fx = fx_new
            else:
                # Try to recover with a smaller step toward the previous point
                x_trial = 0.5 * (x + x_prev)
                fx_trial = call_f(x_trial)
                if fx_trial is not None and abs(fx_trial) < abs(fx):
                    x_prev = x
                    fx_prev = fx
                    x = x_trial
                    fx = fx_trial
                else:
                    break
        
        return x, fx
    
    def brent_like(x0, x1, max_iter=50):
        nonlocal evals
        a = float(x0)
        b = float(x1)
        
        fa = call_f(a)
        fb = call_f(b)
        
        if fa is None or fb is None:
            if fa is not None and fb is None:
                return a, fa
            elif fb is not None and fa is None:
                return b, fb
            else:
                return a, None
        
        if is_good(a, fa):
            return a, fa
        if is_good(b, fb):
            return b, fb
        
        if fa * fb > 0:
            return secant(a, b)
        
        c = a
        fc = fa
        d = b - a
        e = d
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            
            if is_good(b, fb):
                return b, fb
            
            if fa * fb > 0:
                a = c
                fa = fc
                d = b - a
                e = d
            
            if abs(fc) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa
            
            tol1 = 2 * 2.22e-16 * abs(b)
            delta = 0.5 * (c - a)
            
            if abs(delta) <= tol1 or fb == 0:
                return b, fb
            
            if abs(e) < tol1 or abs(fa) <= abs(fb):
                e = delta
                d = fb / fa
            else:
                S = fa / fc
                T1 = fb / fc
                T2 = fb / fa
                d = T1 * (T1 - S) * (b - c) / (fa - fc)
                S = T1 * T2 * (c - a) / (fa - fc)
                if ((d > 0.5 * delta * (c - a)) or (d < 0.5 * delta * (b - c))):
                    d = delta
                if ((c - b) if (d >= 0) else (b - c)) < (3 * tol1 / 2):
                    if abs(d) < tol1:
                        d = delta
            
            e = d
            b = b + d
            fb = call_f(b)
            if fb is None:
                break
        
        return b, fb

    def polish(x0, fx0, max_iter=30):
        """High-precision polish pass using very tight Newton steps."""
        nonlocal evals
        if fx0 is None:
            return x0, fx0
        if is_good(x0, fx0):
            return x0, fx0
        
        x = float(x0)
        fx = fx0
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 1:
                break
            
            # Use central difference with adaptive h for high precision
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = call_f(x + h)
            fx_minus = call_f(x - h)
            if fx_plus is None or fx_minus is None:
                break
            
            d = (fx_plus - fx_minus) / (2 * h)
            if not math.isfinite(d) or abs(d) < 1e-20:
                # Try a larger step
                h = 1e-5 * max(1.0, abs(x))
                fx_plus = call_f(x + h)
                fx_minus = call_f(x - h)
                if fx_plus is None or fx_minus is None:
                    break
                d = (fx_plus - fx_minus) / (2 * h)
                if not math.isfinite(d) or abs(d) < 1e-20:
                    break
            
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            
            if is_good(x_new, fx_new):
                return x_new, fx_new
            
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Try a damped step
                x_trial = x - 0.5 * fx / d
                fx_trial = call_f(x_trial)
                if fx_trial is not None and abs(fx_trial) < abs(fx):
                    x = x_trial
                    fx = fx_trial
                else:
                    break
        
        return x, fx

    # Strategy 1: Newton-Raphson
    x_res, f_res = newton_raphson(x0)
    if is_good(x_res, f_res):
        return x_res
    
    # Strategy 2: Secant with two nearby points
    h = 1e-4 * max(1.0, abs(x0))
    x_res, f_res = secant(x0, x0 + h)
    if is_good(x_res, f_res):
        return x_res
    
    # Strategy 3: Secant with two far points
    h = 1.0
    x_res, f_res = secant(x0 - h, x0 + h)
    if is_good(x_res, f_res):
        return x_res
    
    # Strategy 4: Brent-like with wider interval
    h = 10.0
    x_res, f_res = brent_like(x0 - h, x0 + h)
    if is_good(x_res, f_res):
        return x_res
    
    # Strategy 5: Multi-start random restart to escape local minima
    import random
    rng = random.Random(42)  # Fixed seed for reproducibility
    base_scale = max(1.0, abs(x0))
    
    for attempt in range(10):
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
            break
        
        # Random restart with increasing scale
        scale = base_scale * (10 ** (attempt * 0.5))
        offset = rng.uniform(-scale, scale)
        x_start = x0 + offset
        
        # Try Newton from this point
        x_res, f_res = newton_raphson(x_start)
        if is_good(x_res, f_res):
            return x_res
        
        # Try secant from this point
        h_small = 1e-4 * max(1.0, abs(x_start))
        x_res, f_res = secant(x_start, x_start + h_small)
        if is_good(x_res, f_res):
            return x_res
    
    # Strategy 6: If we have a good candidate but not quite at tolerance, try polish
    if best_f is not None and best_abs < 1e-6:
        x_polished, f_polished = polish(best_x, best_f)
        if is_good(x_polished, f_polished):
            return x_polished
        if f_polished is not None and abs(f_polished) < best_abs:
            best_x = x_polished
            best_f = f_polished
            best_abs = abs(f_polished)
    
    # Strategy 7: Final attempt with a very tight polish on the best candidate
    if best_f is not None and best_abs < 1e-4:
        x_polished, f_polished = polish(best_x, best_f)
        if is_good(x_polished, f_polished):
            return x_polished
    
    # If all fail, return best x found
    return best_x

# EVOLVE-BLOCK-END
