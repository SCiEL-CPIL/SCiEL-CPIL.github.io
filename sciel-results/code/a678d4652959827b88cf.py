"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def time_up():
        return time.time() >= deadline

    x0 = float(x0)
    
    # Track best solution found
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        ax = abs(fx)
        if ax < best_abs:
            best_x = x
            best_f = fx
            best_abs = ax

    # If already good enough
    if best_abs <= 1e-8:
        return best_x

    # Step 1: Find a bracket [a, b] where f(a)*f(b) < 0
    # Start with a small interval around x0 and expand
    a = x0
    fa = best_f
    
    # Try a few local steps first to see if we can find a sign change nearby
    # This helps if x0 is very close to a root but not exactly on it
    h = 1e-6
    for i in range(10):
        b = x0 + h
        fb = f(b)
        update_best(b, fb)
        if fa * fb < 0:
            break
        a = b
        fa = fb
        h *= 2
        if time_up():
            return best_x
    else:
        # If no sign change found in local expansion, try expanding further
        # We already have a, fa from the last iteration
        # But we might have missed a bracket if the function is flat or oscillating
        # Let's do a more systematic expansion if we didn't break
        # Actually, the loop above breaks if sign change is found.
        # If it finishes without break, we have a, fa at the end.
        # We need to find b such that sign changes.
        # Let's restart expansion logic more robustly if the above didn't find it quickly
        # The above loop expands h exponentially. If it didn't find a sign change in 10 steps,
        # we might be in a region where f doesn't change sign or is very flat.
        # Let's try a wider search.
        
        # Reset a to x0
        a = x0
        fa = f(a)
        update_best(a, fa)
        
        # Expand in both directions
        step = 1.0
        found_bracket = False
        for i in range(50):
            # Try positive direction
            b = x0 + step
            fb = f(b)
            update_best(b, fb)
            if fa * fb < 0:
                found_bracket = True
                break
            # Try negative direction
            b = x0 - step
            fb = f(b)
            update_best(b, fb)
            if fa * fb < 0:
                found_bracket = True
                break
            step *= 2
            if step > 1e10:
                break
            if time_up():
                return best_x
        
        if not found_bracket:
            # No bracket found, return best guess
            return best_x

    # Step 2: Refine the root using a hybrid Secant/Bisection method
    # We have [a, b] with f(a)*f(b) < 0
    # Initialize c to the point closer to zero or midpoint
    # Let's use a standard Brent-like or Illinois/Regula Falsi approach
    # For simplicity and robustness, let's use a modified Regula Falsi with bisection fallback
    
    # Ensure a < b for consistency in some logs, though not strictly necessary for math
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # Main iteration loop
    for _ in range(200):
        if time_up():
            return best_x
        
        # Check convergence
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a
        
        # Secant step
        if fa == fb:
            # Degenerate case, bisect
            c = (a + b) / 2
        else:
            c = b - fb * (b - a) / (fb - fa)
        
        # If c is not strictly inside (a, b), or if it's too close to a or b, bisect
        if c <= a or c >= b:
            c = (a + b) / 2
        else:
            # Check if secant step is "too small" or if it doesn't reduce the interval enough
            # This is a heuristic to prevent slow convergence in ill-conditioned cases
            # If the new interval is not significantly smaller, bisect
            if abs(c - a) < 1e-12 * abs(a) and abs(c - b) < 1e-12 * abs(b):
                c = (a + b) / 2
            # Also, if the secant point is very close to one of the endpoints, it might be slow
            # But usually, if it's inside, it's okay. Let's stick to the basic inside check first.
        
        fc = f(c)
        update_best(c, fc)
        
        if abs(fc) <= 1e-8:
            return c
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
        # Optional: Illinois method modification
        # If the same endpoint is retained, halve the function value at that endpoint
        # This is tricky to implement cleanly without tracking history.
        # Let's stick to simple bisection fallback if secant fails to progress.
        
        # Check for stagnation: if interval size doesn't decrease, bisect next time?
        # The bisection fallback above handles the case where secant goes outside.
        # But what if secant stays inside but moves very little?
        # We can track previous interval length.
        
    return best_x
# EVOLVE-BLOCK-END
