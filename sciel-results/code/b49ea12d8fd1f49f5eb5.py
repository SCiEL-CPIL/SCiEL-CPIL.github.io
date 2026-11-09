"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding using a hybrid strategy:
    1. Local search via Golden Section Search on |f(x)| to find a local minimum.
    2. Refinement using a tighter interval and direct evaluation.
    This avoids derivative calculations (noisy in ill-conditioned problems) 
    and is more efficient than pure bisection when the function is locally 
    monotonic or has a clear local minimum of |f(x)|.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    
    # Golden section constant
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi = 1.0 / phi

    # Wrapper to count evaluations and handle exceptions
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count += 1
        try:
            val = f(x)
            if val is None or math.isnan(val) or math.isinf(val):
                return None
            return val
        except Exception:
            return None

    # Initial evaluation
    x = float(x0)
    fx = safe_f(x)
    if fx is None:
        return x
    if abs(fx) <= tol:
        return x

    best_x, best_abs_f = x, abs(fx)

    # Helper to update best
    def update_best(x_val, f_val):
        nonlocal best_x, best_abs_f
        if f_val is not None:
            abs_val = abs(f_val)
            if abs_val < best_abs_f:
                best_x = x_val
                best_abs_f = abs_val

    # Phase 1: Determine an interval [a, b] that likely contains a root or local min of |f|
    # We start with a small window and expand if the minimum is at the boundary.
    
    # Initial window size
    # Heuristic: if x0 is 0, use 1.0, else scale with x0
    scale = max(1.0, abs(x))
    half_width = 0.1 * scale
    
    # Try to find a local minimum of |f(x)| in [x - half_width, x + half_width]
    # If the min is at the edge, expand the window.
    
    a = x - half_width
    b = x + half_width
    
    # To avoid re-evaluating x, we can use it, but golden section is cleaner with new points.
    # Let's just run golden section on [a, b].
    
    # We will run a limited number of golden section steps to find a local min.
    # If the min is at the boundary, we expand.
    
    # Max expansions
    max_expansions = 10
    current_a, current_b = a, b
    
    for expansion_iter in range(max_expansions):
        if time.time() > deadline or eval_count >= max_evals:
            break
            
        # Perform Golden Section Search on [current_a, current_b]
        # We want to find the x that minimizes |f(x)|
        
        # Initialize golden section points
        # c = a + (1 - 1/phi) * (b - a) = a + phi * (b-a) ? No.
        # Standard: 
        # c = a + (1 - 1/phi)*(b-a)
        # d = a + (1/phi)*(b-a)
        # Since 1/phi = phi - 1? No. phi = (sqrt(5)-1)/2 ~ 0.618.
        # 1/phi = 1.618.
        # Let's use:
        # c = b - (b-a)/phi  (closer to b)
        # d = a + (b-a)/phi  (closer to a)
        # Wait, standard golden section for minimization:
        # c = a + (1 - 1/phi)*(b-a)
        # d = a + (1/phi)*(b-a)
        # 1 - 1/phi = 1 - 1.618 = -0.618? No.
        # phi = 0.618...
        # 1/phi = 1.618...
        # c = a + (1 - 1/phi)*(b-a) is negative offset.
        
        # Correct Golden Section:
        # Let r = phi = (sqrt(5)-1)/2 ~ 0.618
        # c = a + (1 - r)*(b - a)
        # d = a + r*(b - a)
        # This places c and d symmetrically.
        
        r = phi
        c = current_a + (1.0 - r) * (current_b - current_a)
        d = current_a + r * (current_b - current_a)
        
        fc = safe_f(c)
        fd = safe_f(d)
        
        if fc is None or fd is None:
            break
            
        update_best(c, fc)
        update_best(d, fd)
        
        # Check if we already have a root
        if abs(fc) <= tol:
            return c
        if abs(fd) <= tol:
            return d
            
        # Iterate to narrow down the interval
        # We run up to 50 iterations of golden section here
        for _ in range(50):
            if time.time() > deadline or eval_count >= max_evals:
                break
                
            # Compare |f(c)| and |f(d)|
            abs_fc = abs(fc)
            abs_fd = abs(fd)
            
            if abs_fc < abs_fd:
                # Minimum is in [current_a, d]
                current_b = d
                b = d
                d = c
                fd = fc
                # New c
                c = current_a + (1.0 - r) * (current_b - current_a)
                fc = safe_f(c)
                if fc is None:
                    break
                update_best(c, fc)
                if abs(fc) <= tol:
                    return c
            else:
                # Minimum is in [c, current_b]
                current_a = c
                a = c
                c = d
                fc = fd
                # New d
                d = current_a + r * (current_b - current_a)
                fd = safe_f(d)
                if fd is None:
                    break
                update_best(d, fd)
                if abs(fd) <= tol:
                    return d
            
            # Convergence check
            if abs(current_b - current_a) < 1e-12 * max(1.0, abs((current_a + current_b)/2)):
                break
        
        # After golden section, the best point is in [current_a, current_b]
        # Check if the best point is at the boundary
        # If best_x is close to current_a or current_b, we might need to expand.
        
        # Estimate the center
        center = (current_a + current_b) / 2.0
        
        # If the best point found so far is very close to the boundary, expand
        # We use a threshold: if best_x is within 10% of the interval width from the edge
        width = current_b - current_a
        if best_x <= current_a + 0.1 * width or best_x >= current_b - 0.1 * width:
            # Expand the interval
            # Double the width, centered on the current best_x or center?
            # Centering on best_x is safer to keep the min inside.
            new_half_width = width # Double the half-width? 
            # Current width is W. New width 2W.
            # Let's center on best_x
            new_a = best_x - width
            new_b = best_x + width
            # Ensure we don't shrink
            if new_a < current_a or new_b > current_b:
                current_a = new_a
                current_b = new_b
            else:
                break
        else:
            # We found a local minimum inside
            break

    # Phase 2: Refinement
    # We have a small interval [current_a, current_b] containing the local min of |f(x)|.
    # The root is where f(x)=0. 
    # If f(current_a) and f(current_b) have opposite signs, use Bisection.
    # If not, the minimum of |f| is the best approximation.
    
    fa = safe_f(current_a)
    fb = safe_f(current_b)
    
    if fa is None or fb is None:
        return best_x
        
    update_best(current_a, fa)
    update_best(current_b, fb)
    
    if abs(fa) <= tol:
        return current_a
    if abs(fb) <= tol:
        return current_b
        
    if fa * fb < 0:
        # Bracket found, use bisection for high precision
        lo, hi = current_a, current_b
        f_lo, f_hi = fa, fb
        
        for _ in range(100):
            if time.time() > deadline or eval_count >= max_evals:
                break
                
            mid = (lo + hi) / 2.0
            f_mid = safe_f(mid)
            if f_mid is None:
                break
            update_best(mid, f_mid)
            
            if abs(f_mid) <= tol:
                return mid
            
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
                
            if hi - lo < 1e-14 * max(1.0, abs(mid)):
                break
                
        return (lo + hi) / 2.0
    else:
        # No sign change, return the best point found (local min of |f|)
        return best_x

# EVOLVE-BLOCK-END
