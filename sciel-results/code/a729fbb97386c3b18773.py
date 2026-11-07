"""Root finder for 3 fixed oscillatory equations using global multi-scale bracketing + Illinois refinement."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    MAX_CALLS = 490     # Stay well under 500

    calls = 0
    best_x = float(x0)
    best_abs_f = float('inf')

    def _f(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= MAX_CALLS:
            return float('nan')
        if time.time() - start_time > TIME_LIMIT:
            return float('nan')
        calls += 1
        try:
            val = f(x)
        except:
            return float('nan')
        if not math.isfinite(val):
            return float('nan')
        abs_val = abs(val)
        if abs_val < best_abs_f:
            best_abs_f = abs_val
            best_x = x
        return val

    def illinois(fa, a, fb, b):
        """Illinois method with safeguarded convergence."""
        nonlocal calls
        for _ in range(200):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            
            # Illinois: linear interpolation with damping
            if fa == fb:
                mid = (a + b) / 2.0
            else:
                mid = (a * fb - b * fa) / (fb - fa)
            
            # Safeguard: if outside bracket or non-finite, use midpoint
            if not (a < mid < b) or not math.isfinite(mid):
                mid = (a + b) / 2.0
            
            fm = _f(mid)
            if math.isnan(fm):
                break
            if abs(fm) <= 1e-8:
                return mid
            
            if fa * fm < 0:
                b = mid
                fb = fm
                fa = fa / 2.0
            else:
                a = mid
                fa = fm
                fb = fb / 2.0
        
        # Return the better endpoint
        if abs(fa) <= abs(fb):
            return a
        return b

    # Phase 1: Global multi-scale grid search to find best bracket
    x = float(x0)
    fx = _f(x)
    if math.isnan(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Collect candidate brackets from multi-scale exploration
    brackets = []  # List of (fa, a, fb, b, score) where score = max(|fa|, |fb|)
    
    # Multi-scale exploration: try different step sizes and ranges
    base_scale = max(1.0, abs(x))
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    
    for scale in scales:
        if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
            break
        
        step = scale * base_scale
        
        # Explore in positive direction
        prev_x = x
        prev_fx = fx
        for i in range(1, 15):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            cur_x = x + i * step
            cur_fx = _f(cur_x)
            if math.isnan(cur_fx):
                break
            if abs(cur_fx) <= 1e-8:
                return cur_x
            if prev_fx * cur_fx < 0:
                score = max(abs(prev_fx), abs(cur_fx))
                brackets.append((prev_fx, prev_x, cur_fx, cur_x, score))
            prev_x = cur_x
            prev_fx = cur_fx
        
        # Explore in negative direction
        prev_x = x
        prev_fx = fx
        for i in range(1, 15):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            cur_x = x - i * step
            cur_fx = _f(cur_x)
            if math.isnan(cur_fx):
                break
            if abs(cur_fx) <= 1e-8:
                return cur_x
            if prev_fx * cur_fx < 0:
                score = max(abs(prev_fx), abs(cur_fx))
                brackets.append((prev_fx, prev_x, cur_fx, cur_x, score))
            prev_x = cur_x
            prev_fx = cur_fx
    
    # Phase 2: If no brackets found, do a finer local scan
    if not brackets:
        local_step = 0.005 * base_scale
        prev_x = x - 20 * local_step
        prev_fx = _f(prev_x)
        if not math.isnan(prev_fx) and abs(prev_fx) <= 1e-8:
            return prev_x
        
        for i in range(41):
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            cur_x = x - 20 * local_step + i * local_step
            cur_fx = _f(cur_x)
            if math.isnan(cur_fx):
                break
            if abs(cur_fx) <= 1e-8:
                return cur_x
            if not math.isnan(prev_fx) and prev_fx * cur_fx < 0:
                score = max(abs(prev_fx), abs(cur_fx))
                brackets.append((prev_fx, prev_x, cur_fx, cur_x, score))
            prev_x = cur_x
            prev_fx = cur_fx
    
    # Phase 3: Select the best bracket and refine with Illinois
    if brackets:
        # Sort by score (smaller is better - closer to root)
        brackets.sort(key=lambda b: b[4])
        
        # Try the best bracket first, then others if needed
        for fa, a, fb, b, _ in brackets:
            if calls >= MAX_CALLS or time.time() - start_time > TIME_LIMIT:
                break
            result = illinois(fa, a, fb, b)
            if result is not None and math.isfinite(result):
                return result
    
    # Return best x found
    return best_x

# EVOLVE-BLOCK-END
