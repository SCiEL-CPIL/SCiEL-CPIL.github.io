import math
import time
import scipy.optimize as opt

def find_root(f, x0):
    START_TIME = time.time()
    TIME_LIMIT = 145.0  # Safety margin before 150s hard limit
    MAX_CALLS = 495
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

    # Evaluate initial point
    fx0 = safe_f(x0)
    if found:
        return best_x

    # Helper to check if two points form a bracket
    def is_bracket(xa, xb, fa, fb):
        if not math.isfinite(fa) or not math.isfinite(fb):
            return False
        return fa * fb < 0

    # Refine a bracket to high precision using bisection with secant acceleration
    def refine_root(a, b, fa, fb):
        nonlocal best_x, best_f, found
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Bisection loop with secant acceleration
        for _ in range(50):
            if found:
                return best_x
            
            # Try secant step if it stays within bracket
            if math.isfinite(fa) and math.isfinite(fb) and fa != fb:
                x_sec = a - fa * (b - a) / (fb - fa)
                if a < x_sec < b:
                    f_sec = safe_f(x_sec)
                    if math.isfinite(f_sec):
                        if abs(f_sec) <= 1e-8:
                            return x_sec
                        if is_bracket(a, x_sec, fa, f_sec):
                            b, fb = x_sec, f_sec
                        elif is_bracket(x_sec, b, f_sec, fb):
                            a, fa = x_sec, f_sec
                        else:
                            # Secant didn't bracket, fall through to bisection
                            pass
                    if found:
                        return best_x
                    if time.time() - START_TIME > TIME_LIMIT:
                        break
            
            # Bisection step
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            if not math.isfinite(fm):
                break
            
            if abs(fm) <= 1e-8:
                return mid

            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        
        # Return the best candidate
        candidates = [a, b]
        if 'mid' in locals():
            candidates.append(mid)
        if 'x_sec' in locals():
            candidates.append(x_sec)
        
        best_local = min(candidates, key=lambda x: abs(cache.get(x, safe_f(x))))
        return best_local

    # Unified geometric scan strategy
    # Start with fine steps near x0, then expand with geometric growth
    
    prev_x = x0
    prev_fx = fx0
    
    # Phase 1: Fine local scan +/- 1.0 with step 0.01
    h = 0.01
    for i in range(1, 101):
        x_new = x0 + i * h
        fx_new = safe_f(x_new)
        if found:
            return best_x
        if is_bracket(prev_x, x_new, prev_fx, fx_new):
            return refine_root(prev_x, x_new, prev_fx, fx_new)
        prev_x = x_new
        prev_fx = fx_new
        
    prev_x = x0
    prev_fx = fx0
    for i in range(1, 101):
        x_new = x0 - i * h
        fx_new = safe_f(x_new)
        if found:
            return best_x
        if is_bracket(x_new, prev_x, fx_new, prev_fx):
            return refine_root(x_new, prev_x, fx_new, prev_fx)
        prev_x = x_new
        prev_fx = fx_new

    # Phase 2: Geometric expansion
    # Continue scanning outward with increasing step sizes
    pos_frontier = x0 + 1.0
    neg_frontier = x0 - 1.0
    pos_fx = cache.get(pos_frontier, safe_f(pos_frontier))
    neg_fx = cache.get(neg_frontier, safe_f(neg_frontier))
    
    if found:
        return best_x

    # Geometric steps: 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4
    steps = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6]
    
    for step in steps:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        if calls >= MAX_CALLS:
            break
            
        # Positive direction
        x_curr = pos_frontier
        fx_curr = pos_fx
        max_pos = x0 + 5000.0  # Cap at 5000
        
        while x_curr < max_pos and calls < MAX_CALLS:
            if found:
                return best_x
            x_new = x_curr + step
            fx_new = safe_f(x_new)
            if is_bracket(x_curr, x_new, fx_curr, fx_new):
                return refine_root(x_curr, x_new, fx_curr, fx_new)
            x_curr = x_new
            fx_curr = fx_new
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
        pos_frontier = x_curr
        pos_fx = fx_curr
        
        # Negative direction
        x_curr = neg_frontier
        fx_curr = neg_fx
        min_neg = x0 - 5000.0
        
        while x_curr > min_neg and calls < MAX_CALLS:
            if found:
                return best_x
            x_new = x_curr - step
            fx_new = safe_f(x_new)
            if is_bracket(x_new, x_curr, fx_new, fx_curr):
                return refine_root(x_new, x_curr, fx_new, fx_curr)
            x_curr = x_new
            fx_curr = fx_new
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
        neg_frontier = x_curr
        neg_fx = fx_curr

    # If no bracket found, return best_x
    return best_x
