"""Root finding for ill-conditioned equations: Global Residual Minimization via Nelder-Mead."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin below 120s limit
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
                return float('inf') # Treat non-finite as high cost for minimization
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs:
                best_f = val
                best_x = x
                best_abs = abs_val
            return val
        except Exception:
            return float('inf')

    # Objective function for minimization: |f(x)|
    def objective(x):
        val = safe_eval(x)
        if val is None:
            return float('inf')
        return abs(val) if val is not None else float('inf')

    # 1. Initial Check
    if best_f is not None and abs(best_f) <= 1e-8:
        return best_x

    # Helper: Local Refinement using Bisection/Regula Falsi
    # This is used when we have a bracket [a, b] with a sign change, or when we are very close.
    def refine_local(a, b, fa, fb, max_iters=100):
        nonlocal eval_count, best_x, best_f, best_abs
        
        if fa is None or fb is None:
            return best_x
            
        # If no sign change, we can't use bisection directly.
        # However, if we are just polishing, we might not have a sign change.
        # This function is primarily for the case where we HAVE a sign change.
        if fa * fb > 0:
            # No sign change. Return the point with smaller |f|
            if abs(fa) < abs(fb):
                return a
            else:
                return b

        for _ in range(max_iters):
            if check_budget():
                break
                
            # Regula Falsi step
            if fa == fb:
                c = (a + b) / 2.0
            else:
                c = (a * fb - b * fa) / (fb - fa)
                
            # Ensure c is within bracket
            if c <= a or c >= b:
                c = (a + b) / 2.0
            
            fc = safe_eval(c)
            if fc is None:
                break
            
            if abs(fc) <= 1e-8:
                return c
            
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
                
            if abs(b - a) < 1e-12:
                return (a + b) / 2.0
        
        return best_x

    # Helper: Nelder-Mead Simplex Method for 1D
    # We minimize g(x) = |f(x)|
    def nelder_mead_1d(x_start, initial_step, max_iters=200):
        nonlocal best_x, best_f, best_abs
        
        # Simplex vertices
        x1 = x_start
        x2 = x_start + initial_step
        
        f1 = objective(x1)
        f2 = objective(x2)
        
        # Ensure x1 is the better point (lower objective)
        if f2 < f1:
            x1, x2 = x2, x1
            f1, f2 = f2, f1
            
        if f1 <= 1e-8:
            return x1

        for _ in range(max_iters):
            if check_budget():
                break
                
            # If the simplex has collapsed, stop
            if abs(x2 - x1) < 1e-12:
                break
                
            # Generate new point
            # Reflection: x_r = x1 - alpha * (x2 - x1)
            alpha = 1.0
            x_r = x1 - alpha * (x2 - x1)
            f_r = objective(x_r)
            
            if f_r <= 1e-8:
                return x_r

            if f_r < f1:
                # Reflection is better than x1
                # Expansion?
                gamma = 2.0
                x_e = x1 - gamma * (x2 - x1)
                f_e = objective(x_e)
                
                if f_e <= 1e-8:
                    return x_e
                    
                if f_e < f_r:
                    x1, f1 = x_e, f_e
                else:
                    x1, f1 = x_r, f_r
            else:
                # Reflection is worse than x1
                if f_r < f2:
                    # Reflection is better than x2 but worse than x1
                    x2, f2 = x_r, f_r
                else:
                    # Contraction
                    beta = 0.5
                    x_c = x1 + beta * (x2 - x1)
                    f_c = objective(x_c)
                    
                    if f_c <= 1e-8:
                        return x_c
                    
                    if f_c < f2:
                        x2, f2 = x_c, f_c
                    else:
                        # Shrink
                        # Move x2 towards x1
                        x2 = x1 + 0.5 * (x2 - x1)
                        f2 = objective(x2)
                        
                        if f2 <= 1e-8:
                            return x2
                        
                        # If we are stuck shrinking, try a random restart or larger jump
                        # For 1D, if we keep shrinking, we might be in a local min of |f| that isn't 0.
                        # But since we are minimizing |f|, a local min of |f| is often a root or a point where f crosses 0 but we missed the sign change due to numerical issues?
                        # Actually, if f is continuous, |f| has a cusp at the root. Nelder-Mead should handle this.
                        # If we are stuck, break and let global search handle it.
                        if abs(x2 - x1) < 1e-10:
                            break

        return x1

    # Main Strategy:
    # 1. Try a few Nelder-Mead runs with different initial step sizes.
    # 2. After each run, check if we found a root.
    # 3. If we found a bracket (sign change) during the process, refine with Bisection.
    
    # We don't easily capture sign changes during NM because we only track best |f|.
    # So we will rely on NM to get us close to a root, then try to find a bracket around the best point.
    
    steps = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    for step in steps:
        if check_budget():
            break
            
        # Run Nelder-Mead
        x_candidate = nelder_mead_1d(best_x, step, max_iters=50)
        
        if x_candidate is not None:
            f_cand = safe_eval(x_candidate)
            if f_cand is not None and abs(f_cand) <= 1e-8:
                return x_cand
            
            # Try to find a bracket around the best point found so far
            # We scan locally around best_x to find a sign change
            h = step * 0.5
            found_bracket = False
            a, b = None, None
            fa, fb = None, None
            
            # Scan a small range around best_x
            # We already have best_x and best_f
            # Let's check neighbors
            x_left = best_x - h
            x_right = best_x + h
            
            f_left = safe_eval(x_left)
            f_right = safe_eval(x_right)
            
            if f_left is not None and f_right is not None:
                if best_f * f_left < 0:
                    a, b = x_left, best_x
                    fa, fb = f_left, best_f
                    found_bracket = True
                elif best_f * f_right < 0:
                    a, b = best_x, x_right
                    fa, fb = best_f, f_right
                    found_bracket = True
                elif f_left * f_right < 0:
                    a, b = x_left, x_right
                    fa, fb = f_left, f_right
                    found_bracket = True
            
            if found_bracket:
                x_refined = refine_local(a, b, fa, fb, max_iters=100)
                f_ref = safe_eval(x_refined)
                if f_ref is not None and abs(f_ref) <= 1e-8:
                    return x_refined

    # If NM didn't find a root, try a broader random search / multi-start
    # Since NM is local, we need to kick it out of local minima.
    # Let's try a few random perturbations and run NM from there.
    import random
    random.seed(42) # Deterministic for reproducibility in testing, though evaluator might vary
    
    for _ in range(10):
        if check_budget():
            break
        
        # Random perturbation
        perturbation = random.uniform(-10, 10)
        x_start = best_x + perturbation
        
        # Run NM with a moderate step
        x_candidate = nelder_mead_1d(x_start, 1.0, max_iters=30)
        
        if x_candidate is not None:
            f_cand = safe_eval(x_candidate)
            if f_cand is not None and abs(f_cand) <= 1e-8:
                return x_cand

    # Final fallback: Return best found
    return best_x
# EVOLVE-BLOCK-END
