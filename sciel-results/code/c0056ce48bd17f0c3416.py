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
                # Try a damped secant step
                x_trial = x - 0.5 * fx * (x - x_prev) / denom
                fx_trial = call_f(x_trial)
                if fx_trial is not None and abs(fx_trial) < abs(fx):
                    x_prev = x
                    fx_prev = fx
                    x = x_trial
                    fx = fx_trial
                else:
                    # Try even smaller
                    x_trial2 = x - 0.25 * fx * (x - x_prev) / denom
                    fx_trial2 = call_f(x_trial2)
                    if fx_trial2 is not None and abs(fx_trial2) < abs(fx):
                        x_prev = x
                        fx_prev = fx
                        x = x_trial2
                        fx = fx_trial2
                    else:
                        break
        
        return x, fx

    def brent_like(x0, x1, max_iter=50):
        nonlocal evals
        # Ensure f(a)*f(b) < 0 if possible, otherwise just use secant/newton hybrid
        a = float(x0)
        b = float(x1)
        
        fa = call_f(a)
        fb = call_f(b)
        
        if fa is None or fb is None:
            # Return best known
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
        
        # If no sign change, just use secant
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
            
            # Invariant: fa*fb < 0
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
                # Bisection
                e = delta
                d = fb / fa
            else:
                # Inverse quadratic interpolation
                S = fa / fc
                T1 = fb / fc
                T2 = fb / fa
                d = T1 * (T1 - S) * (b - c) / (fa - fc)
                S = T1 * T2 * (c - a) / (fa - fc)
                # Check if d is in range
                if ((d > 0.5 * delta * (c - a)) or (d < 0.5 * delta * (b - c))):
                    d = delta
                if ((c - b) if (d >= 0) else (b - c)) < (3 * tol1 / 2):
                    d = delta * ((c - b) if (fb >= 0) else (b - c)) / abs(fb)
                    if d < 0:
                        d = -d
                    else:
                        d = d
                    # Actually just use bisection if too small
                    if abs(d) < tol1:
                        d = delta
            
            e = d
            b = b + d
            fb = call_f(b)
            if fb is None:
                break
        
        return b, fb

    # Strategy 1: Newton-Raphson with adaptive derivative
    x_res, f_res = newton_raphson(x0)
    if is_good(x_res, f_res):
        return x_res
    
    # Strategy 2: Secant with nearby points (multiple scales)
    for h_scale in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        h = h_scale * max(1.0, abs(x0))
        if h < 1e-12:
            continue
        x_res, f_res = secant(x0, x0 + h)
        if is_good(x_res, f_res):
            return x_res
    
    # Strategy 3: Try Newton from slightly perturbed starting points
    for pert in [1e-3, -1e-3, 1e-2, -1e-2]:
        x_pert = x0 + pert * max(1.0, abs(x0))
        x_res, f_res = newton_raphson(x_pert)
        if is_good(x_res, f_res):
            return x_res
    
    # Strategy 4: Brent-like with various bracket widths
    for h in [1.0, 5.0, 10.0, 50.0, 100.0]:
        x_res, f_res = brent_like(x0 - h, x0 + h)
        if is_good(x_res, f_res):
            return x_res
    
    # Strategy 5: Try secant with asymmetric brackets
    for h in [1.0, 10.0, 100.0]:
        x_res, f_res = secant(x0 - h, x0 + 2*h)
        if is_good(x_res, f_res):
            return x_res
    
    # Strategy 6: Final high-precision polish pass
    # If we're close (|f| < 1e-6), try to tighten with more iterations
    if best_f is not None and abs(best_f) < 1e-6:
        # Try Newton polish from best_x
        x_res, f_res = newton_raphson(best_x, max_iter=30)
        if is_good(x_res, f_res):
            return x_res
        
        # Try secant polish
        h = 1e-4 * max(1.0, abs(best_x))
        x_res, f_res = secant(best_x, best_x + h, max_iter=30)
        if is_good(x_res, f_res):
            return x_res
    
    # If all fail, return best x found
    return best_x

# EVOLVE-BLOCK-END
