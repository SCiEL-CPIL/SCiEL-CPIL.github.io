"""
Strategy: Global Multi-Scale Sign-Change Sweep with Illinois Refinement
1. Global Search: Sample f at points on a logarithmic scale expanding from x0.
   This ensures we cover a vast domain efficiently and find a bracket [a, b] 
   where f(a) and f(b) have opposite signs.
2. Local Refinement: Once a bracket is found, use the Illinois Method.
   The Illinois Method is a hybrid of Bisection and Secant methods.
   It maintains the bracket (guaranteeing convergence) but converges much 
   faster than pure bisection by weighting the midpoint towards the side 
   with the smaller function value. This is robust against oscillations 
   and does not require derivatives.
3. Budget Management: Strictly limit the number of function calls (e.g., 400) 
   to maximize the score component (1 - 0.5 * log10(mean_calls) / log10(500)).
   400 calls is close to the limit of 500 but allows for a high success rate 
   while keeping the penalty low.
"""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin for 120s limit
    MAX_CALLS = 450     # Keep well below 500 to maintain high score
    TOL = 1e-9          # Target tolerance
    
    calls = 0
    cache = {}
    
    def _f(x):
        nonlocal calls
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError("Time limit exceeded")
        
        # Check cache
        # Use a rounded key to handle float precision issues in cache lookups
        # But for exact roots, we want exact keys if possible.
        # Let's just use the float directly. Python floats are hashable.
        key = x
        if key in cache:
            return cache[key]
            
        if calls >= MAX_CALLS:
            # If we hit the limit, we can't make new calls.
            # We should return a dummy or raise? 
            # Raising will fail the equation. 
            # Better to return the last known value? No, that's dangerous.
            # We should check budget before calling.
            raise RuntimeError("Call limit exceeded")
            
        calls += 1
        val = f(x)
        cache[key] = val
        return val

    # State for best result found so far
    best_x = float('nan')
    best_f_val = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_f_val
        if math.isfinite(x) and math.isfinite(fx):
            if abs(fx) < abs(best_f_val):
                best_x = x
                best_f_val = fx

    # 1. Initial check
    try:
        fx0 = _f(x0)
        if abs(fx0) <= TOL:
            return x0
        update_best(x0, fx0)
    except Exception:
        # If initial call fails, we have no info. Return x0? 
        # The problem says f takes/returns float. Exceptions from f are not expected 
        # except maybe domain errors? "non-finite x" is a failure condition for the 
        # *result*, but if f raises, we should handle it.
        # Assuming f is well-behaved for finite inputs.
        return x0

    # 2. Global Search for a Bracket
    # We search in both directions from x0.
    # We use a logarithmic scale to cover large ranges quickly.
    
    points = [(x0, fx0)]
    bracket = None
    
    # Define search directions: +1 and -1
    # We'll try to find a sign change between consecutive points.
    
    # Initial step size
    # If x0 is 0, start with 1.0. Otherwise, relative step.
    # We'll use a mix of absolute and relative steps.
    
    # Offsets to try
    # We want to cover [x0-1e6, x0+1e6] roughly, but logarithmically.
    # Steps: 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5
    
    # If x0 is large, scale the offsets.
    scale = max(1.0, abs(x0))
    
    # Base offsets
    base_offsets = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    
    # Scale them
    offsets = [s * scale for s in base_offsets]
    
    # Also try 0.0 if x0 is not close to 0
    if abs(x0) > 1e-3:
        offsets.append(0.0 - x0) # This is the offset to reach 0. 
        # Wait, my offset logic is x0 + offset.
        # So offset to reach 0 is -x0.
        
    # Sort offsets by absolute value to prioritize local search?
    # No, let's just iterate.
    
    # We already have x0.
    # Let's evaluate the offsets.
    
    # We need to be careful about calls.
    # Max calls 450.
    # We have used 1.
    # We can afford ~200 points for global search.
    # 9 offsets * 2 directions = 18 calls. Very cheap.
    
    # Let's expand the list of points.
    # points list contains (x, fx)
    
    for s in offsets:
        # Positive direction
        xp = x0 + s
        if calls < MAX_CALLS - 20: # Leave budget for refinement
            try:
                fxp = _f(xp)
                update_best(xp, fxp)
                points.append((xp, fxp))
                # Check for bracket immediately with new point and all previous
                # To keep it fast, just check with the last few?
                # No, check all. N is small.
                for (xq, fxq) in points[:-1]:
                    if fxp * fxq < 0:
                        bracket = (xp, xq)
                        break
            except Exception:
                pass
            
            if bracket:
                break
        
        # Negative direction
        xn = x0 - s
        if calls < MAX_CALLS - 20:
            try:
                fxn = _f(xn)
                update_best(xn, fxn)
                points.append((xn, fxn))
                for (xq, fxq) in points[:-1]:
                    if fxn * fxq < 0:
                        bracket = (xn, xq)
                        break
            except Exception:
                pass
            
            if bracket:
                break

    # If no bracket found, try a denser local grid around x0?
    if not bracket:
        # Try 100 points in a small window around x0
        # Window size: max(1e-3, abs(x0)*1e-3)
        win = max(1e-3, abs(x0)*1e-3)
        a = x0 - win
        b = x0 + win
        # We already have some points. Let's just do a simple linear sweep of 50 points.
        # This costs 50 calls.
        if calls < MAX_CALLS - 100:
            step = (b - a) / 50.0
            prev_x = a
            try:
                prev_fx = _f(prev_x)
                update_best(prev_x, prev_fx)
                for i in range(1, 51):
                    cur_x = a + i * step
                    cur_fx = _f(cur_x)
                    update_best(cur_x, cur_fx)
                    if prev_fx * cur_fx < 0:
                        bracket = (prev_x, cur_x)
                        break
                    prev_x = cur_x
                    prev_fx = cur_fx
            except Exception:
                pass

    # 3. Refinement using Illinois Method
    if bracket:
        a, b = bracket
        # We need f(a) and f(b)
        # They are in cache or we can retrieve them.
        # But we need to ensure they are in the cache.
        # If they were evaluated in the loop above, they are in cache.
        # If not, we need to call _f.
        
        try:
            fa = _f(a)
            fb = _f(b)
        except Exception:
            fa = None
            fb = None
        
        if fa is not None and fb is not None and fa * fb < 0:
            # Illinois Method
            # We maintain weighted values fa_w, fb_w
            fa_w = fa
            fb_w = fb
            
            # Max iterations for refinement
            # Each iteration costs 1 call.
            # We have ~400 calls left.
            # Illinois converges fast. 100 iterations is plenty.
            
            for _ in range(100):
                if time.time() - start_time > TIME_LIMIT:
                    break
                if calls >= MAX_CALLS:
                    break
                    
                # Calculate weighted midpoint
                # m = (a * |fb_w| + b * |fa_w|) / (|fa_w| + |fb_w|)
                denom = abs(fa_w) + abs(fb_w)
                if denom == 0:
                    break
                m = (a * abs(fb_w) + b * abs(fa_w)) / denom
                
                try:
                    fm = _f(m)
                except Exception:
                    break
                    
                update_best(m, fm)
                
                if abs(fm) <= TOL:
                    return m
                    
                if fa * fm < 0:
                    # Root is in [a, m]
                    b = m
                    fb = fm
                    fb_w = fm / 2.0
                else:
                    # Root is in [m, b]
                    a = m
                    fa = fm
                    fa_w = fm / 2.0
                    
                # Check interval width
                if abs(b - a) < 1e-12:
                    break
            
            # Return the best of a, b, m
            # We should check which one has the smallest |f|
            # We have fa, fb, fm (last one)
            # But we might have updated best_x in update_best.
            # Let's just return the one with smallest |f| from the last few.
            candidates = [(a, fa), (b, fb), (m, fm)]
            best_cand = min(candidates, key=lambda x: abs(x[1]))
            return best_cand[0]

    # If no bracket found, return best_x found
    if math.isfinite(best_x):
        return best_x
    return x0

# EVOLVE-BLOCK-END
