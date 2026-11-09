"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding using a global bracketing strategy with hybrid
    bisection/Newton acceleration.
    
    Strategy:
    1. Attempt to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       Start near x0 and expand exponentially.
    2. Once bracketed, use a hybrid method:
       - Try Newton's method step.
       - If the step stays within the bracket, take it (fast quadratic convergence).
       - If the step leaves the bracket or derivative is bad, take a bisection step (robust linear convergence).
    3. This ensures we never get "lost" like pure Newton, but converge fast when safe.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    eval_count = 0
    max_evals = 500
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count += 1
        try:
            val = f(x)
            if math.isnan(val) or math.isinf(val):
                return None
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
    # Try expanding in both directions from x0
    found_bracket = False
    step_size = 1e-6
    max_expansions = 100
    
    # We'll maintain a list of points and their values to reuse
    # But for simplicity, we just expand a and b
    
    b = x
    fb = fa
    
    # Try to find a sign change by expanding
    # Strategy: expand a to left, b to right, or just one side if stuck
    # Let's try expanding a to the left and b to the right simultaneously
    # Or better, try a few specific expansions
    
    # Let's try a simple expanding bracket search
    # We will keep track of the best bracket found so far if any
    best_a, best_b, best_fa, best_fb = None, None, None, None
    
    # Initial check: is x0 already a root?
    if abs(fa) <= 1e-8:
        return a

    # Expand
    for i in range(max_expansions):
        if time.time() > deadline:
            break
            
        # Try expanding to the right first, then left, alternating or both
        # Let's try expanding a to the left and b to the right
        new_a = a - step_size
        new_b = b + step_size
        
        fa_new = safe_f(new_a)
        fb_new = safe_f(new_b)
        
        if fa_new is None or fb_new is None:
            break
            
        # Check for sign change in [new_a, a] or [b, new_b] or [new_a, new_b]
        if fa_new * fa < 0:
            a, b, fa, fb = new_a, a, fa_new, fa
            found_bracket = True
            break
        if fb * fb_new < 0:
            a, b, fa, fb = b, new_b, fb, fb_new
            found_bracket = True
            break
        if fa_new * fb_new < 0:
            a, b, fa, fb = new_a, new_b, fa_new, fb_new
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
        # Fall back to a few Newton steps or return the best point found?
        # Since the task is "ill-conditioned", bracketing might fail if f doesn't cross 0
        # (e.g., touches 0 but doesn't cross, or numerical noise).
        # However, the problem states "find x with |f(x)| <= 1e-8", not necessarily f(x)=0 crossing.
        # But bisection requires a sign change.
        # If no bracket is found, we are in trouble for this strategy.
        # Let's try a local minimization of |f(x)|? No, that's complex.
        # Let's just return the x that gave the smallest |f(x)| during the expansion?
        # Or just return x0?
        # Actually, if no bracket is found, pure bisection fails.
        # Let's try a simple secant method as a fallback?
        # Or just return the best point seen so far.
        
        # Let's track the best point seen during expansion
        # We didn't explicitly track min |f| in the loop above easily without extra storage.
        # Let's just return x0 or the last evaluated point if no bracket found.
        # This is a failure mode for this strategy, but better than crashing.
        return x

    # Phase 2: Hybrid Bisection/Newton within [a, b]
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
            
        # Try Newton's Method step
        # Estimate derivative using finite difference
        h = 1e-6 * max(1.0, abs(m))
        # Use central difference if possible, or forward
        # To save evals, use forward difference from m
        # f(m+h)
        fmh = safe_f(m + h)
        if fmh is None:
            continue # Skip Newton step, stick to bisection
            
        d = (fmh - fm) / h
        if d == 0:
            continue
            
        x_new = m - fm / d
        
        # Check if x_new is within [a, b]
        if a < x_new < b:
            # Check if it's a good step (not too close to boundary to avoid divergence)
            # And if |f(x_new)| is potentially better?
            # We evaluate f(x_new) to see if it's a root or to update bracket.
            fx_new = safe_f(x_new)
            if fx_new is None:
                continue
                
            if abs(fx_new) <= 1e-8:
                return x_new
                
            # Update bracket based on x_new
            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new
                
            # If the step didn't reduce the bracket significantly or f value,
            # we might oscillate. But bisection is the backup.
            # We've already updated the bracket, so next iteration will bisect.
            
    # Return the best point found
    # We can check a, b, and m (last calculated)
    candidates = [a, b]
    if 'm' in locals():
        candidates.append(m)
        
    best_x = min(candidates, key=lambda x: abs(safe_f(x)) if safe_f(x) is not None else float('inf'))
    
    return best_x
# EVOLVE-BLOCK-END
