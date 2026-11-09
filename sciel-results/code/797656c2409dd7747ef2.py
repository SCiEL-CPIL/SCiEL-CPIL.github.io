"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust hybrid approach:
    1. Attempt to find a sign change bracket via expansion.
    2. If bracket found, use Safeguarded Secant/Bisection.
    3. If no bracket, use Golden Section Search to minimize |f(x)|, then polish with Secant.
    """
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    max_evals = 500
    eval_count = [0]
    best_x = x0
    best_f_val = float('inf')
    
    def eval_f(x):
        nonlocal best_x, best_f_val
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        eval_count[0] += 1
        try:
            val = f(x)
            if math.isnan(val) or math.isinf(val):
                return None
            # Track best
            abs_val = abs(val)
            if abs_val < best_f_val:
                best_f_val = abs_val
                best_x = x
            return val
        except (ValueError, OverflowError, ZeroDivisionError):
            return None

    # Helper: Check if we have a root
    def is_root(x, fx):
        return fx is not None and abs(fx) <= 1e-8

    # Strategy 1: Find a bracket [a, b] with sign change
    # Start with small steps and expand
    a = x0
    fa = eval_f(a)
    if fa is None:
        return best_x
    if is_root(a, fa):
        return a

    # Try to find a sign change
    h = 1e-3 * max(1.0, abs(x0))
    b = x0 + h
    fb = eval_f(b)
    
    bracket_found = False
    if fb is not None and fa * fb < 0:
        bracket_found = True
    else:
        # Expand search range
        for _ in range(20):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            # Double the step size, alternate direction if needed
            h *= 2.0
            # Try x0 + h and x0 - h
            candidates = [x0 + h, x0 - h]
            for xc in candidates:
                fc = eval_f(xc)
                if fc is None:
                    continue
                if is_root(xc, fc):
                    return xc
                # Check sign change with current best known point 'a'
                if fa * fc < 0:
                    a, fa = a, fa
                    b, fb = xc, fc
                    bracket_found = True
                    break
                # Also check if it changes sign with previous b? 
                # Simpler: just keep expanding from x0
            if bracket_found:
                break

    if bracket_found:
        # Safeguarded Secant / Bisection
        # We have [a, b] with fa*fb < 0
        # Use Secant, but if step leaves bracket, use Bisection
        x_prev = a
        x_curr = b
        f_prev = fa
        f_curr = fb
        
        for _ in range(100):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            if is_root(x_curr, f_curr):
                return x_curr
            if is_root(x_prev, f_prev):
                return x_prev
            
            # Secant step
            if f_curr == f_prev:
                # Degenerate, bisect
                x_new = 0.5 * (x_prev + x_curr)
            else:
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            # Safeguard: if x_new is outside [min(a,b), max(a,b)], use bisection
            a_min, a_max = min(a, b), max(a, b)
            if x_new < a_min or x_new > a_max:
                x_new = 0.5 * (a_min + a_max)
            
            f_new = eval_f(x_new)
            if f_new is None:
                # If eval failed, shrink bracket? Or break.
                # Let's try to bisect the current known valid points
                # But we don't have a valid midpoint eval. 
                # Just break and return best.
                break
            
            if is_root(x_new, f_new):
                return x_new
            
            # Update bracket
            if f_prev * f_new < 0:
                # Root is in [x_prev, x_new]
                x_curr, f_curr = x_new, f_new
            else:
                # Root is in [x_new, x_curr]
                x_prev, f_prev = x_new, f_new
            
            # Update global bracket [a, b] for safeguarding
            if f_prev * f_new < 0:
                 # This logic is slightly tricky with sliding window.
                 # Let's just maintain the two most recent points that bracket the root.
                 pass

        # After loop, return best
        return best_x

    else:
        # No bracket found. Minimize |f(x)| using Golden Section Search
        # Define a local search range
        # We have evaluated x0, and some neighbors. 
        # Let's use the best point found so far as center.
        center = best_x
        h = 1e-2 * max(1.0, abs(center))
        
        # Initial interval [center - h, center + h]
        a = center - h
        b = center + h
        
        # Ensure we have function values for a and b
        fa = eval_f(a)
        fb = eval_f(b)
        
        # If evals failed, shrink or return best
        if fa is None or fb is None:
            return best_x
            
        # Golden Section Search for min |f(x)|
        # We want to minimize g(x) = |f(x)|
        # Golden ratio
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = eval_f(c)
        fd = eval_f(d)
        
        # Handle None
        if fc is None: fc = float('inf')
        if fd is None: fd = float('inf')
        
        for _ in range(100):
            if time.time() > deadline or eval_count[0] >= max_evals:
                break
            
            # Determine which half to keep
            if abs(fc) < abs(fd):
                # Min is in [a, d]
                b = d
                fb = fd
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_f(c)
                if fc is None:
                    fc = float('inf')
            else:
                # Min is in [c, b]
                a = c
                fa = fc
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_f(d)
                if fd is None:
                    fd = float('inf')
            
            # Check for root in current best
            if abs(fc) <= 1e-8:
                return c
            if abs(fd) <= 1e-8:
                return d
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b

            # Convergence check
            if abs(b - a) < 1e-10 * max(1.0, abs(a)):
                break
        
        # Polish the best point found in minimization with Secant
        # Use the best point from minimization as x0 for secant
        x_best_min = best_x
        # Find a second point for secant
        h = 1e-6 * max(1.0, abs(x_best_min))
        x2 = x_best_min + h
        f2 = eval_f(x2)
        f1 = eval_f(x_best_min) # Re-eval to be sure, or use cached? 
        # eval_f updates best_x, so f1 might be stale if best_x changed. 
        # Let's just use the current best_x and its value if available, or re-eval.
        # To save evals, we can try to use the last known values.
        # But for safety, let's do a few secant steps.
        
        if f1 is not None and f2 is not None:
             x_a = x_best_min
             x_b = x2
             f_a = f1
             f_b = f2
             
             for _ in range(20):
                 if time.time() > deadline or eval_count[0] >= max_evals:
                     break
                 if is_root(x_b, f_b):
                     return x_b
                 if is_root(x_a, f_a):
                     return x_a
                 
                 if f_b == f_a:
                     break
                 
                 x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
                 f_new = eval_f(x_new)
                 if f_new is None:
                     break
                 
                 if is_root(x_new, f_new):
                     return x_new
                 
                 # Keep the two points with smallest |f|? Or standard secant update?
                 # Standard secant: x_a, x_b = x_b, x_new
                 # But for ill-conditioned, keeping the best might be better.
                 # Let's stick to standard secant but with damping if step is huge.
                 if abs(x_new - x_b) > 10 * max(1.0, abs(x_b)):
                     x_new = x_b + 0.5 * (x_b - x_a)
                     f_new = eval_f(x_new)
                     if f_new is None:
                         break
                     if is_root(x_new, f_new):
                         return x_new
                 
                 x_a, f_a = x_b, f_b
                 x_b, f_b = x_new, f_new
        
        return best_x

# EVOLVE-BLOCK-END
