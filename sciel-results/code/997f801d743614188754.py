import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using a hybrid Secant/Bisection strategy.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = float(x0)
    best_x = x
    best_f = f(x)
    calls = 1
    
    def check_time():
        return time.time() > deadline

    if abs(best_f) <= 1e-8:
        return best_x

    # Phase 1: Local Secant Search
    # We need two points to start Secant.
    # Use a small step to get the second point.
    scale = max(1.0, abs(x))
    h = 1e-4 * scale
    
    x_prev = x
    f_prev = best_f
    
    x_curr = x + h
    f_curr = f(x_curr)
    calls += 1
    best_x, best_f = (x_curr, f_curr) if abs(f_curr) < abs(best_f) else (best_x, best_f)
    
    if abs(f_curr) <= 1e-8:
        return x_curr

    # Iterate Secant
    max_secant_steps = 20
    for i in range(max_secant_steps):
        if check_time():
            return best_x
        if calls >= 450:
            return best_x

        # Check for convergence
        if abs(f_curr) < abs(f_prev):
            best_x, best_f = x_curr, f_curr
            if abs(f_curr) <= 1e-8:
                return x_curr

        # Secant step
        if f_curr == f_prev:
            # Parallel lines, break to global search
            break
        
        denom = f_curr - f_prev
        if denom == 0:
            break
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        # Safeguard: if step is too large, abort local search
        # "Too large" relative to current position or initial scale
        step_size = abs(x_new - x_curr)
        if step_size > 10.0 * max(1.0, abs(x_curr)) or not math.isfinite(x_new):
            break
            
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        f_curr = f(x_curr)
        calls += 1
        
        if abs(f_curr) < abs(best_f):
            best_x, best_f = x_curr, f_curr
            if abs(f_curr) <= 1e-8:
                return x_curr

    # Phase 2: Global Bracket Expansion + Bisection
    # If we are here, local search failed or was interrupted.
    # We need to find a bracket [a, b] with opposite signs.
    
    # Start expansion from the best point found so far
    a = best_x
    fa = best_f
    if fa == 0:
        return a

    # Try expanding in positive and negative directions
    # We will keep track of a bracket if found
    bracket_found = False
    b = None
    fb = None
    
    # Define expansion steps
    # Start with a small step, then grow exponentially
    base_step = 1.0
    max_expansions = 100
    
    # First, try to see if we can bracket with the points we already have
    # We have x_prev, f_prev and x_curr, f_curr from the loop (or initial)
    # But they might be far apart.
    
    # Let's do a systematic expansion from 'a'
    # We check a small neighborhood first
    directions = [1, -1]
    
    for direction in directions:
        if check_time() or calls >= 450:
            break
        if bracket_found:
            break
            
        step = 1.0
        for j in range(max_expansions):
            if check_time() or calls >= 450:
                break
            
            b_candidate = a + direction * step
            fb_candidate = f(b_candidate)
            calls += 1
            
            if not math.isfinite(fb_candidate):
                step *= 2
                continue
            
            if abs(fb_candidate) < abs(best_f):
                best_x, best_f = b_candidate, fb_candidate
                if abs(fb_candidate) <= 1e-8:
                    return b_candidate

            if fa * fb_candidate < 0:
                # Found a bracket
                bracket_found = True
                b = b_candidate
                fb = fb_candidate
                break
            
            # If the function value is getting huge, maybe we jumped over a root 
            # or went to infinity. Shrink step? No, just keep expanding or try other direction.
            # For oscillatory, expanding is usually good until we hit a sign change.
            step *= 2.0
            
            # If step becomes too large, stop this direction
            if step > 1e10:
                break

    if not bracket_found:
        # If no bracket found, return the best point we have
        return best_x

    # Phase 3: Bisection / Illinois on [a, b]
    # Ensure a < b for clarity, though not strictly necessary
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    # Illinois Algorithm
    for k in range(100):
        if check_time() or calls >= 450:
            break
            
        # Check convergence
        if abs(fa) < abs(fb):
            best_x, best_f = a, fa
        else:
            best_x, best_f = b, fb
            
        if abs(best_f) <= 1e-8:
            return best_x

        # Illinois update
        # Weight the new point towards the side with smaller |f|
        if abs(fa) < abs(fb):
            # New point is closer to a
            b_new = a + (b - a) * (fa / (fa - fb))
            # If b_new is too close to a or b, use midpoint
            if abs(b_new - a) < 1e-12 * max(1, abs(a)) or abs(b_new - b) < 1e-12 * max(1, abs(b)):
                b_new = (a + b) / 2
            fb_old = fb
            b = b_new
            fb = f(b)
            calls += 1
            if not math.isfinite(fb):
                # Fallback to bisection
                b = (a + b) / 2
                fb = f(b)
                calls += 1
        else:
            # New point is closer to b
            a_new = b + (a - b) * (fb / (fa - fb))
            if abs(a_new - b) < 1e-12 * max(1, abs(b)) or abs(a_new - a) < 1e-12 * max(1, abs(a)):
                a_new = (a + b) / 2
            fa_old = fa
            a = a_new
            fa = f(a)
            calls += 1
            if not math.isfinite(fa):
                # Fallback to bisection
                a = (a + b) / 2
                fa = f(a)
                calls += 1

        # Check sign change again (should hold, but safety)
        if fa * fb > 0:
            # Something went wrong, fallback to pure bisection
            mid = (a + b) / 2
            fmid = f(mid)
            calls += 1
            if abs(fmid) < abs(best_f):
                best_x, best_f = mid, fmid
                if abs(fmid) <= 1e-8:
                    return mid
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid

    return best_x
# EVOLVE-BLOCK-END
