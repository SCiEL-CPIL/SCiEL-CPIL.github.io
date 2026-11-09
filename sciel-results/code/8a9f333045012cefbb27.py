import math
import time
import scipy.optimize as opt

def find_root(f, x0):
    START_TIME = time.time()
    TIME_LIMIT = 145.0
    MAX_CALLS = 498
    calls = 0
    cache = {}
    best_x = x0
    best_f = float('inf')
    found = False

    def safe_f(x):
        nonlocal calls, best_x, best_f, found
        if found:
            return 0.0
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - START_TIME > TIME_LIMIT:
            return float('nan')
        
        if x in cache:
            val = cache[x]
        else:
            calls += 1
            try:
                val = f(x)
                val = float(val)
            except Exception:
                val = float('nan')
            cache[x] = val
            
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_f:
                best_f = abs_val
                best_x = x
                if best_f <= 1e-8:
                    found = True
        return val

    # Initial evaluation
    fx0 = safe_f(x0)
    if found:
        return best_x

    # Phase 1: Adaptive scan
    scan_configs = [
        (0.02, 1.0),    # Very fine local scan
        (0.05, 5.0),    # Fine local scan
        (0.1, 20.0),    # Medium range
        (0.5, 100.0),   # Wider
        (1.0, 500.0),   # Wide
        (5.0, 2000.0),  # Very wide
    ]
    
    brackets_found = []
    
    for step, max_range in scan_configs:
        if found:
            return best_x
        
        # Positive direction
        prev_x = x0
        prev_fx = fx0
        for i in range(1, int(max_range/step) + 1):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            x_new = x0 + i * step
            fx_new = safe_f(x_new)
            if found:
                return best_x
            
            if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
                bracket = (prev_x, x_new)
                if not any(abs(bracket[0] - b[0]) < step*0.1 and abs(bracket[1] - b[1]) < step*0.1 for b in brackets_found):
                    brackets_found.append(bracket)
                    result = refine_root(bracket, safe_f, cache)
                    if result is not None:
                        return result
            
            prev_x = x_new
            prev_fx = fx_new
        
        # Negative direction
        prev_x = x0
        prev_fx = fx0
        for i in range(1, int(max_range/step) + 1):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            x_new = x0 - i * step
            fx_new = safe_f(x_new)
            if found:
                return best_x
            
            if math.isfinite(prev_fx) and math.isfinite(fx_new) and prev_fx * fx_new < 0:
                bracket = (x_new, prev_x)
                if not any(abs(bracket[0] - b[0]) < step*0.1 and abs(bracket[1] - b[1]) < step*0.1 for b in brackets_found):
                    brackets_found.append(bracket)
                    result = refine_root(bracket, safe_f, cache)
                    if result is not None:
                        return result
            
            prev_x = x_new
            prev_fx = fx_new
    
    # If no bracket found, return best_x
    return best_x

def refine_root(bracket, safe_f, cache):
    a, b = bracket
    
    # Ensure a < b
    if a > b:
        a, b = b, a
    
    fa = cache.get(a, safe_f(a))
    fb = cache.get(b, safe_f(b))
    
    if not math.isfinite(fa) or not math.isfinite(fb):
        return None
    
    # Try brentq first for fast convergence
    try:
        root = opt.brentq(safe_f, a, b, xtol=1e-15, rtol=1e-15, maxiter=200)
        if math.isfinite(root):
            fr = safe_f(root)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                return root
    except Exception:
        pass
    
    # Fallback: Bisection with tracking of best point
    local_best_x = (a + b) / 2.0
    local_best_f = float('inf')
    
    for _ in range(100):
        mid = (a + b) / 2.0
        fm = safe_f(mid)
        if not math.isfinite(fm):
            break
            
        abs_fm = abs(fm)
        if abs_fm < local_best_f:
            local_best_f = abs_fm
            local_best_x = mid
            
        if abs_fm <= 1e-8:
            return mid
            
        if fa * fm <= 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    
    # Return the best point found
    if local_best_f <= 1e-8:
        return local_best_x
    
    # Check endpoints one more time
    candidates = [(a, abs(fa)), (b, abs(fb)), (local_best_x, local_best_f)]
    best = min(candidates, key=lambda x: x[1])
    if best[1] <= 1e-8:
        return best[0]
    
    return None
