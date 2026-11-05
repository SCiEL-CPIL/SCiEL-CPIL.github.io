"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 140.0  # Safety margin
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val

    def newton_raphson(x0, max_iters=30):
        x = float(x0)
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            fx = call_f(x)
            if fx is None:
                return None
            if abs(fx) <= TOL:
                return x
            h = 1e-7 * max(1.0, abs(x))
            fx2 = call_f(x + h)
            if fx2 is None:
                return None
            d = (fx2 - fx) / h
            if d == 0 or not math.isfinite(d):
                return None
            x_new = x - fx / d
            if not math.isfinite(x_new):
                return None
            # Damping
            step = x_new - x
            max_step = 10 * max(1.0, abs(x))
            if abs(step) > max_step:
                x_new = x + max_step * (1 if step > 0 else -1)
            x = x_new
        return None
    
    def secant(x0, max_iters=30):
        x0 = float(x0)
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        f0 = call_f(x0)
        f1 = call_f(x1)
        if f0 is None or f1 is None:
            return None
        if abs(f0) <= TOL: return x0
        if abs(f1) <= TOL: return x1
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not math.isfinite(x2):
                break
            f2 = call_f(x2)
            if f2 is None:
                break
            if abs(f2) <= TOL:
                return x2
            x0, x1 = x1, x2
            f0, f1 = f1, f2
        return None

    def muller(x0, max_iters=30):
        x = float(x0)
        h = 1e-4 * max(1.0, abs(x))
        x1 = x + h
        x2 = x1 + h
        fx = call_f(x)
        f1 = call_f(x1)
        f2 = call_f(x2)
        if fx is None or f1 is None or f2 is None:
            return None
        if abs(fx) <= TOL: return x
        if abs(f1) <= TOL: return x1
        if abs(f2) <= TOL: return x2
        
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            
            h0 = x1 - x2
            h1 = x - x1
            if h0 == 0 or h1 == 0:
                break
            
            d0 = (f1 - f2) / h0
            d1 = (fx - f1) / h1
            a = (d1 - d0) / (h1 + h0)
            b = a * (h1 + h0) + d0
            c = f2
            
            # Choose the root closer to x2
            # x_new = x2 - 2c / (b + sign(b)*sqrt(b^2 - 4ac))
            disc = b*b - 4*a*c
            if disc < 0:
                # Complex roots, use real part or fallback
                # Try bisection-like step or break
                break
            sq_disc = math.sqrt(disc)
            denom = b + (1 if b >= 0 else -1) * sq_disc
            if denom == 0:
                break
            x_new = x2 - 2*c / denom
            
            if not math.isfinite(x_new):
                break
            
            f_new = call_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new
            
            # Update points: keep the three points closest to the new estimate or with smallest function values
            # Simple update: shift window
            x, fx = x1, f1
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
            
            # Convergence check
            if abs(x_new - x2) < 1e-14 * max(1.0, abs(x_new)):
                 break
                 
        return None

    def find_bracket(x0, fx0, max_expansions=50):
        if fx0 is None:
            return None
        if abs(fx0) <= TOL:
            return (x0, x0, fx0, fx0)
        
        step = 0.1 * max(1.0, abs(x0))
        for direction in (1.0, -1.0):
            b = x0 + direction * step
            fb = call_f(b)
            if fb is None:
                continue
            if fx0 * fb < 0.0:
                if direction > 0:
                    return (x0, b, fx0, fb)
                else:
                    return (b, x0, fb, fx0)
            
            cur = b
            curf = fb
            for _ in range(max_expansions):
                if time.monotonic() > DEADLINE:
                    return None
                step *= 2.0
                nxt = x0 + direction * step
                fn = call_f(nxt)
                if fn is None:
                    break
                if curf * fn < 0.0:
                    if direction > 0:
                        return (cur, nxt, curf, fn)
                    else:
                        return (nxt, cur, fn, curf)
                cur, curf = nxt, fn
        return None

    def illinois(lo, hi, flo, fhi, max_iters=80):
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if flo * fhi > 0.0:
                break
            # Ensure |flo| < |fhi| for Illinois update
            if abs(flo) < abs(fhi):
                lo, hi = hi, lo
                flo, fhi = fhi, flo
            
            denom = fhi - flo
            if abs(denom) < 1e-300:
                mid = 0.5 * (lo + hi)
            else:
                mid = (flo * hi - fhi * lo) / denom
                if not (lo < mid < hi):
                    mid = 0.5 * (lo + hi)
            
            fm = call_f(mid)
            if fm is None:
                break
            if abs(fm) <= TOL:
                return mid
            
            if flo * fm < 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, 0.5 * flo
            
            if abs(hi - lo) <= 1e-14 * max(1.0, abs(mid)):
                return mid
        return None

    # 1. Try local methods from x0
    result = newton_raphson(x0)
    if result is not None: return result
    result = secant(x0)
    if result is not None: return result
    result = muller(x0)
    if result is not None: return result

    # 2. Try bracketing + Illinois
    fx0 = call_f(float(x0))
    bracket = find_bracket(float(x0), fx0)
    if bracket is not None:
        lo, hi, flo, fhi = bracket
        result = illinois(lo, hi, flo, fhi)
        if result is not None: return result

    # 3. Try local methods from offset points
    offsets = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0]
    for off in offsets:
        if time.monotonic() > DEADLINE or evals >= MAX_EVALS:
            break
        x_try = float(x0) + off
        result = newton_raphson(x_try, max_iters=15)
        if result is not None: return result
        result = secant(x_try, max_iters=15)
        if result is not None: return result
        result = muller(x_try, max_iters=15)
        if result is not None: return result
        
        # Try bracketing from offset
        fx_try = call_f(x_try)
        if fx_try is not None:
            bracket = find_bracket(x_try, fx_try, max_expansions=20)
            if bracket is not None:
                lo, hi, flo, fhi = bracket
                result = illinois(lo, hi, flo, fhi, max_iters=40)
                if result is not None: return result

    # 4. Random restarts if budget allows
    if evals < MAX_EVALS - 50 and time.monotonic() < DEADLINE - 10:
        random.seed(42) # Deterministic for reproducibility in scoring if needed, but random is fine
        for _ in range(10):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
                break
            # Sample in a wide range around x0
            scale = max(1.0, abs(float(x0)))
            x_try = float(x0) + random.uniform(-10*scale, 10*scale)
            
            # Quick check
            fx_try = call_f(x_try)
            if fx_try is None:
                continue
            if abs(fx_try) <= TOL:
                return x_try
            
            # Try local
            result = newton_raphson(x_try, max_iters=10)
            if result is not None: return result
            result = secant(x_try, max_iters=10)
            if result is not None: return result
            
            # Try bracket
            bracket = find_bracket(x_try, fx_try, max_expansions=10)
            if bracket is not None:
                lo, hi, flo, fhi = bracket
                result = illinois(lo, hi, flo, fhi, max_iters=20)
                if result is not None: return result

    return best_x
# EVOLVE-BLOCK-END
