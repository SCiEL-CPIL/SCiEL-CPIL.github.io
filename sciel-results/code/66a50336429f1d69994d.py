"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding using the Illinois Algorithm.
    
    Strategy:
    1. Find a bracket [a, b] where f(a) and f(b) have opposite signs.
       Use a geometric expansion from x0.
    2. Use the Illinois Algorithm (modified bisection) to converge.
       - Unlike pure bisection, it moves the midpoint closer to the point 
         with the smaller |f|, assuming the function is roughly linear.
       - This typically converges faster than bisection (superlinear in practice 
         for smooth functions) but remains globally convergent like bisection.
       - It is more robust than Newton/secant for ill-conditioned problems 
         because it never leaves the bracket and does not require derivatives.
    3. Handle edge cases: no bracket found, numerical errors, timeout.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    eval_count = 0
    max_evals = 500
    
    # Cache for f evaluations
    cache = {}

    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        
        # Simple cache check
        if x in cache:
            return cache[x]
        
        eval_count += 1
        try:
            val = f(x)
            if math.isnan(val) or math.isinf(val):
                return None
            cache[x] = val
            return val
        except Exception:
            return None

    x = float(x0)
    
    # Phase 1: Find a bracket [a, b]
    # We need f(a) and f(b) to have opposite signs.
    
    a = x
    fa = safe_f(a)
    
    if fa is None:
        return x
        
    if abs(fa) <= 1e-8:
        return a

    # Expand to find a sign change
    # We'll try expanding in both directions, but keep track of the best bracket.
    # Actually, a simpler robust strategy:
    # Start with a small interval [x0 - delta, x0 + delta].
    # If no sign change, expand delta.
    
    delta = 1e-3
    max_delta = 1e6
    found_bracket = False
    b = None
    fb = None
    
    # Try expanding right first, then left, or both?
    # Let's try a spiral: 
    # 1. Check [x0, x0+delta]
    # 2. Check [x0-delta, x0]
    # 3. Increase delta
    
    for _ in range(50):
        if time.time() > deadline or eval_count >= max_evals:
            break
            
        # Try right side
        b_right = x + delta
        fb_right = safe_f(b_right)
        if fb_right is not None:
            if fa * fb_right < 0:
                a, b, fa, fb = x, b_right, fa, fb_right
                found_bracket = True
                break
            # If we didn't find a bracket on the right, maybe the root is further right
            # But we should also check if the sign changed relative to the previous right point?
            # For simplicity, let's just expand.
            
        # Try left side
        a_left = x - delta
        fa_left = safe_f(a_left)
        if fa_left is not None:
            if fa_left * fa < 0:
                a, b, fa, fb = a_left, x, fa_left, fa
                found_bracket = True
                break
            
        # If no bracket found with current delta, expand
        delta *= 2.0
        if delta > max_delta:
            break

    # If we still don't have a bracket, try a wider search or fallback
    if not found_bracket:
        # Fallback: return the point with the smallest |f| seen so far
        best_x = x
        best_f = abs(fa)
        for px, pf in cache.items():
            if pf is not None and abs(pf) < best_f:
                best_f = abs(pf)
                best_x = px
        return best_x

    # Phase 2: Illinois Algorithm
    # We have [a, b] with f(a)*f(b) < 0.
    
    # Ensure |fa| <= |fb| initially for standard Illinois setup? 
    # Actually, Illinois works regardless, but the damping factor depends on which side is "smaller".
    # Standard Illinois:
    # 1. Compute midpoint m = (a+b)/2
    # 2. Evaluate f(m)
    # 3. If |f(a)| > |f(b)|:
    #    a = m
    #    fa = f(m)
    #    b = (a + b) / (1 + |fa|/|fb|)  <-- This is the weighted midpoint
    #    Wait, the standard Illinois update is:
    #    If we replace a with m (because |fa| was larger? No, usually we keep the smaller one)
    #    
    # Let's use the standard formulation:
    # Keep a and b such that |fa| <= |fb|.
    # 1. m = (a+b)/2
    # 2. fm = f(m)
    # 3. If |fa| > |fb|: swap a,b and fa,fb. Now |fa| <= |fb|.
    # 4. Check if fm is root.
    # 5. If fa*fm < 0: b=m, fb=fm. Else a=m, fa=fm.
    # 6. If |fa| > |fb|: swap again.
    # 7. Next midpoint: m = (a + b * |fa| / (|fa| + |fb|)) ? 
    #    Actually, the Illinois method replaces the larger function value point with the midpoint, 
    #    and then uses a weighted midpoint for the next step.
    
    # A simpler, robust implementation of Illinois:
    # 1. Start with [a, b] where f(a)*f(b) < 0.
    # 2. While not converged:
    #    a. m = (a + b) / 2
    #    b. fm = f(m)
    #    c. If |fm| <= tol: return m
    #    d. If f(a)*fm < 0: b=m, fb=fm
    #       Else: a=m, fa=fm
    #    e. If |fa| > |fb|: swap a,b and fa,fb
    #    f. Next m is calculated as: m = (a + b * |fa| / (|fa| + |fb|)) 
    #       Wait, this is not quite right. 
    #       
    # Let's stick to the classic Illinois definition:
    # 1. m = (a+b)/2
    # 2. fm = f(m)
    # 3. If f(a)*fm < 0: b=m, fb=fm
    #    Else: a=m, fa=fm
    # 4. If |fa| > |fb|: swap a,b and fa,fb
    # 5. Next midpoint: m = (a + b * |fa| / (|fa| + |fb|))
    #    This biases the next midpoint towards the side with the smaller function value (a).
    
    while time.time() < deadline and eval_count < max_evals:
        # Check convergence of endpoints
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
            
        # Calculate next midpoint
        # If |fa| and |fb| are very different, the weighted average is closer to the smaller one.
        # If they are similar, it's close to the middle.
        # Handle case where |fa|+|fb| is 0 (shouldn't happen if not root)
        denom = abs(fa) + abs(fb)
        if denom == 0:
            m = (a + b) / 2.0
        else:
            # Weighted towards 'a' if |fa| < |fb|
            # m = (a * |fb| + b * |fa|) / (|fa| + |fb|)
            # This is a convex combination.
            # If |fa| is small, weight of b is small, so m is close to a.
            m = (a * abs(fb) + b * abs(fa)) / denom
            
        # Ensure m is strictly inside (a, b) to avoid infinite loops due to float precision
        # If m == a or m == b, fall back to bisection
        if m <= a or m >= b:
            m = (a + b) / 2.0
            if m <= a or m >= b:
                # Interval is too small or stuck
                break

        fm = safe_f(m)
        if fm is None:
            break
            
        if abs(fm) <= 1e-8:
            return m
            
        # Update bracket
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
            
        # Optional: swap to keep |fa| <= |fb| for the weighting formula?
        # The formula m = (a*|fb| + b*|fa|)/(|fa|+|fb|) works regardless of order, 
        # but it biases towards the point with the smaller absolute value.
        # If we don't swap, it still biases correctly.
        
    # Return best point
    candidates = [a, b]
    if 'm' in locals():
        candidates.append(m)
        
    best_x = min(candidates, key=lambda x: abs(safe_f(x)) if safe_f(x) is not None else float('inf'))
    
    return best_x
# EVOLVE-BLOCK-END
