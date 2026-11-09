"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding using a global bracketing strategy with pure bisection.
    
    Strategy:
    1. Attempt to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       Start near x0 and expand exponentially in both directions.
    2. Once bracketed, use pure bisection.
       - This is robust for ill-conditioned problems where derivatives are noisy or zero.
       - Convergence is linear (O(log(1/eps))) but guaranteed for continuous functions with sign change.
    3. If no bracket is found, fall back to returning the best point seen or x0.
    
    This avoids the pitfalls of Newton's method (divergence, derivative failure) 
    and secant method (oscillation in ill-conditioned cases) by relying on 
    the intermediate value theorem.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    eval_count = 0
    max_evals = 500
    
    # Cache for f evaluations to avoid redundant calls
    cache = {}

    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        
        # Check cache
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
    
    # Phase 1: Find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small interval around x0
    
    a = x
    fa = safe_f(a)
    
    if fa is None:
        return x # Can't even evaluate
    
    if abs(fa) <= 1e-8:
        return a

    # Expand search for a bracket
    # Strategy: expand a to left, b to right, or just one side if stuck
    # We will maintain a list of points and their values to reuse
    # But for simplicity, we just expand a and b
    
    b = x
    fb = fa
    
    # Try to find a sign change by expanding
    # Strategy: expand a to the left and b to the right, or just one side if stuck
    # Let's try expanding a to the left and b to the right simultaneously
    # Or better, try a few specific expansions
    
    # Let's try a simple expanding bracket search
    # We will keep track of the best bracket found so far if any
    best_a, best_b, best_fa, best_fb = None, None, None, None
    
    # Initial check: is x0 already a root?
    if abs(fa) <= 1e-8:
        return a

    # Expand
    # Use a larger initial step to catch wider brackets faster
    step_size = 1e-3 
    max_expansions = 100
    
    # We need to be careful with the expansion logic.
    # If we expand both sides, we might miss a bracket if the function is monotonic 
    # and the root is on one side only.
    # Better approach: 
    # 1. Try expanding to the right from x0.
    # 2. If that fails, try expanding to the left from x0.
    # 3. If that fails, try expanding both ways (current logic).
    
    # Actually, a more robust expansion is:
    # Start with [x0, x0+step]. Check sign.
    # If no sign, move to [x0+step, x0+2*step], etc.
    # And also [x0-step, x0], [x0-2*step, x0-step], etc.
    
    # Let's use a "spiral" expansion or just alternate left/right.
    # Simple approach: 
    # Try [x0, x0+step], [x0-step, x0], [x0, x0+2*step], [x0-2*step, x0]...
    
    # Even simpler: 
    # Expand a to the left and b to the right, but check for sign change in 
    # [a, x0] and [x0, b] as we go.
    
    a = x
    b = x
    fa = safe_f(a)
    fb = safe_f(b)
    
    found_bracket = False
    step_size = 1e-3
    
    for i in range(max_expansions):
        if time.time() > deadline:
            break
            
        # Expand a to the left and b to the right
        new_a = a - step_size
        new_b = b + step_size
        
        fa_new = safe_f(new_a)
        fb_new = safe_f(new_b)
        
        if fa_new is None or fb_new is None:
            break
            
        # Check for sign change in [new_a, a] or [b, new_b]
        if fa_new * fa < 0:
            a, b, fa, fb = new_a, a, fa_new, fa
            found_bracket = True
            break
        if fb * fb_new < 0:
            a, b, fa, fb = b, new_b, fb, fb_new
            found_bracket = True
            break
            
        # Update bounds for next expansion
        a, fa = new_a, fa_new
        b, fb = new_b, fb_new
        step_size *= 2.0
        
        # If we've gone too far, maybe the root is not accessible or function is pathological
        if abs(a) > 1e100 or abs(b) > 1e100:
            break

    if not found_bracket:
        # If we didn't find a bracket, we can't guarantee convergence with bisection.
        # Fall back to returning the best point found?
        # Since the task is "ill-conditioned", bracketing might fail if f doesn't cross 0
        # (e.g., touches 0 but doesn't cross, or numerical noise).
        # However, the problem states "find x with |f(x)| <= 1e-8", not necessarily f(x)=0 crossing.
        # But bisection requires a sign change.
        # If no bracket is found, we are in trouble for this strategy.
        # Let's try a simple secant method as a fallback?
        # Or just return the x that gave the smallest |f(x)| during the expansion?
        
        # We didn't explicitly track min |f| in the loop above easily without extra storage.
        # Let's just return x0 or the last evaluated point if no bracket found.
        # This is a failure mode for this strategy, but better than crashing.
        
        # Let's try one more thing: if we have points in the cache, find the one with min |f|
        best_x = x
        best_f = abs(fa)
        for px, pf in cache.items():
            if pf is not None and abs(pf) < best_f:
                best_f = abs(pf)
                best_x = px
        return best_x

    # Phase 2: Bisection within [a, b]
    # We know f(a) and f(b) have opposite signs.
    # We want to find x such that |f(x)| <= 1e-8.
    
    # To save evaluations, we can check if a or b is already good enough
    if abs(fa) <= 1e-8:
        return a
    if abs(fb) <= 1e-8:
        return b

    while (time.time() < deadline and eval_count < max_evals):
        # Calculate midpoint
        m = (a + b) / 2.0
        
        # Check if interval is small enough
        # We don't strictly need the interval to be tiny if |f(m)| is small.
        
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
            
    # Return the best point found
    # We can check a, b, and m (last calculated)
    candidates = [a, b]
    if 'm' in locals():
        candidates.append(m)
        
    best_x = min(candidates, key=lambda x: abs(safe_f(x)) if safe_f(x) is not None else float('inf'))
    
    return best_x
# EVOLVE-BLOCK-END
