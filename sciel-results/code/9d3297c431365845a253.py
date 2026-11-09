"""Root finding for ill-conditioned equations: Global Residual Minimization via Nelder-Mead."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 480   # Safety margin below 500 limit
    
    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    try:
        best_f = f(best_x)
        eval_count += 1
        if math.isfinite(best_f):
            best_abs = abs(best_f)
    except Exception:
        best_f = float('inf')
        best_abs = float('inf')

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                # Return a large penalty value for optimization
                return 1e100
            eval_count += 1
            if abs(val) < best_abs:
                best_f = val
                best_x = x
                best_abs = abs(val)
            return val
        except Exception:
            return 1e100

    # Objective function for minimization: square of f(x)
    def objective(x):
        val = safe_eval(x)
        if val is None:
            return 1e100
        if val == 1e100: # Was a penalty
            return 1e100
        return val * val

    # 1. Initial Check
    if best_f is not None and abs(best_f) <= 1e-8:
        return best_x

    # 2. Nelder-Mead Simplex Method
    # We minimize f(x)^2. 
    # Initial simplex points: x0, x0+delta, x0-delta
    # delta should be adaptive.
    
    # Determine initial step size based on x0 magnitude
    if abs(x0) > 1e-6:
        delta = abs(x0) * 0.01
    else:
        delta = 1e-2

    # Simplex vertices
    # For 1D, a simplex is just 2 points (a line segment). 
    # However, standard Nelder-Mead in 1D is essentially bisection/secant hybrid.
    # To make it "global" and robust, we can use a 3-point simplex in 1D? 
    # No, 1D simplex is 2 points. 
    # Let's implement a robust 1D Nelder-Mead (which is similar to Brent's method but 
    # more flexible in handling non-convexity by reflecting/expanding).
    
    # Actually, for 1D, Brent's method is superior to Nelder-Mead.
    # But Brent's requires a bracket. 
    # Strategy: 
    # 1. Use a "Golden Section" search over a wide interval to find a local minimum of |f(x)|.
    # 2. Once a local minimum is found, check if it's a root.
    # 3. If not, expand the interval and repeat.
    # This is different from the current "multi-start local solver" because it actively
    # searches for the *minimum* of the residual, not just a sign change.
    
    # Let's implement a robust "Minimize Residual" strategy.
    # Phase 1: Find a bracket for the minimum of |f(x)| using expanding intervals.
    # Phase 2: Refine using Golden Section or Brent.
    
    # Since we don't know the domain, we expand intervals around x0.
    # Interval: [x0 - R, x0 + R]
    
    R = 1.0
    max_R = 100.0
    found_bracket = False
    a_bracket = None
    b_bracket = None
    
    # We want to find a region where |f(x)| is smaller than at the boundaries.
    # This indicates a local minimum inside.
    
    while R <= max_R and not check_budget():
        a = x0 - R
        b = x0 + R
        
        fa = safe_eval(a)
        fb = safe_eval(b)
        
        if fa is None or fb is None:
            R *= 2
            continue
            
        # Check if x0 is a better candidate than boundaries (local min indicator)
        # Or if there is a sign change (root indicator)
        if fa * fb < 0:
            # Sign change found, we can use Brent's method directly
            a_bracket = a
            b_bracket = b
            found_bracket = True
            break
        
        # Check if there's a local minimum of |f| inside
        # Simple heuristic: evaluate a few points inside
        # If any point has |f| < min(|fa|, |fb|), we have a candidate minimum region.
        # Let's sample 3 points: x0, x0-R/2, x0+R/2
        f_mid1 = safe_eval(x0 - R/2)
        f_mid2 = safe_eval(x0 + R/2)
        f_center = safe_eval(x0) # Already evaluated, but safe_eval handles caching? No, it re-evaluates. 
        # Optimization: We should cache or avoid re-evaluating if possible. 
        # For now, assume safe_eval is cheap enough or we track it. 
        # Actually, safe_eval increments count. Let's be careful.
        
        # Let's just check if the center is better than boundaries
        if f_center is not None and f_mid1 is not None and f_mid2 is not None:
            if abs(f_center) < abs(fa) and abs(f_center) < abs(fb):
                # x0 is a local minimum of |f|
                # We can try to refine around x0
                a_bracket = x0 - R/2
                b_bracket = x0 + R/2
                found_bracket = True
                break
            elif abs(f_mid1) < abs(fa) and abs(f_mid1) < abs(f_center):
                a_bracket = x0 - R
                b_bracket = x0
                found_bracket = True
                break
            elif abs(f_mid2) < abs(fb) and abs(f_mid2) < abs(f_center):
                a_bracket = x0
                b_bracket = x0 + R
                found_bracket = True
                break

        # If no local min indicator, expand
        R *= 2.0

    # If we found a bracket for a minimum or sign change, refine it.
    if found_bracket and not check_budget():
        # Use Brent's method if sign change, otherwise Golden Section for min |f|
        fa = safe_eval(a_bracket)
        fb = safe_eval(b_bracket)
        
        if fa is None or fb is None:
            return best_x
            
        if fa * fb < 0:
            # Sign change: Use Brent's Method (scipy.optimize.brentq is perfect here)
            # But we must respect eval budget. 
            # SciPy's brentq takes a function. We can wrap it to track evals.
            # However, brentq might use too many evals if not careful.
            # Let's implement a manual Brent-like or just use bisection/secant hybrid 
            # which is safer for budget control.
            
            # Robust Hybrid: Secant with Bisection safeguard (Illinois method)
            a, b = a_bracket, b_bracket
            fa, fb = fa, fb
            
            for _ in range(100):
                if check_budget():
                    break
                
                # Illinois Method: 
                # If fa and fb have opposite signs, root is in [a,b]
                if fa == fb:
                    break
                    
                # Secant step
                c = (a * fb - b * fa) / (fb - fa)
                
                # If c is outside [a,b], use bisection
                if c <= a or c >= b:
                    c = (a + b) / 2.0
                    fc = safe_eval(c)
                else:
                    fc = safe_eval(c)
                
                if fc is None:
                    break
                    
                if abs(fc) <= 1e-8:
                    return c
                
                if fa * fc < 0:
                    b, fb = c, fc
                    # Illinois: dampen the smaller side
                    if abs(fa) < abs(fb):
                        fa = fa / 2.0
                else:
                    a, fa = c, fc
                    if abs(fb) < abs(fa):
                        fb = fb / 2.0
                    
                if abs(b - a) < 1e-12:
                    return (a + b) / 2.0
        else:
            # No sign change, but we bracketed a minimum of |f|
            # Use Golden Section Search to minimize |f(x)|
            gr = (math.sqrt(5) - 1) / 2
            a, b = a_bracket, b_bracket
            fa = abs(fa)
            fb = abs(fb)
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            fc = abs(safe_eval(c)) if safe_eval(c) is not None else float('inf')
            fd = abs(safe_eval(d)) if safe_eval(d) is not None else float('inf')
            
            for _ in range(100):
                if check_budget():
                    break
                
                if fc < fd:
                    b = d
                    fb = fd
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = abs(safe_eval(c)) if safe_eval(c) is not None else float('inf')
                else:
                    a = c
                    fa = fc
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = abs(safe_eval(d)) if safe_eval(d) is not None else float('inf')
                    
                if abs(b - a) < 1e-12:
                    break

            # Return the best point found in this interval
            # We have best_x tracked globally
            # Let's check the midpoint
            mid = (a + b) / 2
            f_mid = safe_eval(mid)
            if f_mid is not None and abs(f_mid) <= 1e-8:
                return mid

    # If all else fails, return best_x found so far
    return best_x
# EVOLVE-BLOCK-END
