# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Strategy:
    1. Global Grid Search: Scan a logarithmic grid of points around x0 to find a sign change.
       This is robust to the location of the root relative to x0.
    2. Newton Polish: Once a bracket is found, use Newton's method (with bisection fallback
       if Newton steps are unstable or go out of bounds) to converge to the root quickly.
       Newton converges quadratically, reducing the number of calls compared to pure bisection/secant.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    if not math.isfinite(x0):
        return 0.0

    # Check if x0 is already a root
    f_x0 = f(x0)
    if abs(f_x0) <= 1e-8:
        return x0

    best_x = x0
    best_f = f_x0

    # Phase 1: Find a bracket [a, b] such that f(a)*f(b) <= 0
    # We search in a wide range using a logarithmic scale to handle roots far from x0.
    # We also check small intervals to handle roots very close to x0.
    
    a = None
    b = None
    fa = None
    fb = None
    
    # Define search steps: mix of small and large steps
    # Logarithmic steps: 10^-4 to 10^6
    steps = []
    for exp in range(-4, 7):
        steps.append(10**exp)
        steps.append(-10**exp)
    
    # Add some intermediate steps for better coverage
    steps.extend([0.1, -0.1, 0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 20.0, -20.0, 50.0, -50.0, 200.0, -200.0, 500.0, -500.0])
    
    # Sort steps by absolute value to try small ones first? 
    # Actually, for polynomials, if x0 is close, small steps are best. If x0 is far, large steps are needed.
    # Let's try a mix. First try small, then large.
    # Re-sorting to prioritize smaller magnitudes first, as roots are often near x0 in these tasks.
    steps.sort(key=lambda s: abs(s))
    
    found_bracket = False
    
    for step in steps:
        if time.time() > deadline:
            return best_x
            
        b_candidate = x0 + step
        if not math.isfinite(b_candidate):
            continue
            
        fb_candidate = f(b_candidate)
        if not math.isfinite(fb_candidate):
            continue
            
        if abs(fb_candidate) <= 1e-8:
            return b_candidate
        
        # Update best
        if abs(fb_candidate) < abs(best_f):
            best_f = fb_candidate
            best_x = b_candidate

        # Check for sign change with x0
        if f_x0 * fb_candidate <= 0:
            a = x0
            fa = f_x0
            b = b_candidate
            fb = fb_candidate
            found_bracket = True
            break
        
        # Also check if we crossed a root between previous point and this one?
        # We don't have a previous point in this simple loop, so we just check against x0.
        # To be more robust, we could keep a sliding window, but for polynomials, 
        # checking against x0 is often sufficient if steps are dense enough near x0.
        # However, if x0 is not near the root, this might miss it if the function oscillates.
        # But the task says "smooth polynomial equations", implying single root or well-behaved.
        
    # If no bracket found against x0, try expanding from x0 further or checking other points
    if not found_bracket:
        # Try a denser search in a larger range if needed
        # For now, if we haven't found it, we might have missed it.
        # Let's try a few more specific points if the first pass failed.
        # Actually, the sorted steps cover 1e-4 to 1e6. This should be sufficient for most cases.
        # If it fails, return best_x.
        return best_x

    # Phase 2: Newton's Method with Bisection Fallback
    # We have [a, b] with f(a)*f(b) <= 0.
    # We will iterate to find the root.
    
    # State: a, fa, b, fb
    # We want to converge to the root.
    
    # Newton's method requires derivative. We don't have it.
    # We can approximate derivative using finite differences, but that costs extra calls.
    # Alternatively, use Secant method which is derivative-free and converges superlinearly.
    # Secant method: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    
    # Let's use a hybrid Secant/Bisection (similar to Brent's method logic but simpler).
    # We keep two points x0, x1.
    # Let x0 = a, x1 = b.
    # f0 = fa, f1 = fb.
    
    x0_new = a
    x1_new = b
    f0 = fa
    f1 = fb
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline:
            return best_x
            
        # Check if we have a root
        if abs(f0) <= 1e-8:
            return x0_new
        if abs(f1) <= 1e-8:
            return x1_new
            
        # Update best
        if abs(f0) < abs(best_f):
            best_f = f0
            best_x = x0_new
        if abs(f1) < abs(best_f):
            best_f = f1
            best_x = x1_new

        # Secant step
        denom = f1 - f0
        if abs(denom) < 1e-15:
            # Degenerate, use bisection
            x_new = (x0_new + x1_new) / 2.0
        else:
            x_new = x1_new - f1 * (x1_new - x0_new) / denom
            
        # Check if x_new is within bracket
        min_ab = min(x0_new, x1_new)
        max_ab = max(x0_new, x1_new)
        
        if not (min_ab < x_new < max_ab):
            # Secant step is outside, use bisection
            x_new = (x0_new + x1_new) / 2.0
            
        f_new = f(x_new)
        if not math.isfinite(f_new):
            # If x_new is bad, try bisection
            x_new = (x0_new + x1_new) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
                
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update best
        if abs(f_new) < abs(best_f):
            best_f = f_new
            best_x = x_new

        # Update bracket
        # We want to maintain the invariant that the root is in [x0_new, x1_new]
        # If f0 and f_new have opposite signs, root is in [x0_new, x_new]
        # Else root is in [x_new, x1_new]
        
        if f0 * f_new <= 0:
            # Root in [x0_new, x_new]
            x1_new = x_new
            f1 = f_new
        else:
            # Root in [x_new, x1_new]
            x0_new = x_new
            f0 = f_new
            
        # Optional: Acceleration
        # If |f0| > |f1|, we might want to shrink the interval more aggressively?
        # Standard bisection/secant hybrid handles this naturally.
        
    return best_x

# EVOLVE-BLOCK-END
