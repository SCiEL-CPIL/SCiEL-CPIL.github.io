"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting from x0 using inverse quadratic interpolation
    with bisection fallback for robustness on ill-conditioned functions.
    
    Returns x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    """
    MAX_EVALS = 500
    TOL = 1e-8
    
    eval_count = 0
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise ValueError("Evaluation limit exceeded")
        eval_count += 1
        return f(x)
    
    x = float(x0)
    fx = safe_f(x)
    if abs(fx) <= TOL:
        return x
    
    # Try to establish a bracket by probing in both directions with expanding steps
    a, b, fa, fb = None, None, None, None
    base_step = 1e-4 * max(1.0, abs(x))
    
    for direction in [1, -1]:
        for scale in [1, 10, 100, 1000, 10000]:
            if eval_count >= MAX_EVALS:
                break
            x_try = x + direction * base_step * scale
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if fx * fx_try < 0:
                if fx < fx_try:
                    a, b = x, x_try
                    fa, fb = fx, fx_try
                else:
                    a, b = x_try, x
                    fa, fb = fx_try, fx
                break
        if a is not None:
            break
    
    # If we found a bracket, use IQI with bisection fallback
    if a is not None:
        # History for IQI: maintain last 3 distinct points
        history = [(a, fa), (b, fb)]
        
        for _ in range(MAX_EVALS - eval_count):
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            
            new_x = None
            
            # Try IQI if we have 3 distinct points with distinct f values
            if len(history) >= 3:
                x1, f1 = history[-3]
                x2, f2 = history[-2]
                x3, f3 = history[-1]
                
                # Check for distinct f values (needed for IQI)
                if abs(f1 - f2) > 1e-14 and abs(f2 - f3) > 1e-14 and abs(f1 - f3) > 1e-14:
                    denom1 = (f1 - f2) * (f1 - f3)
                    denom2 = (f2 - f1) * (f2 - f3)
                    denom3 = (f3 - f1) * (f3 - f2)
                    
                    if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                        term1 = x1 * (0 - f2) * (0 - f3) / denom1
                        term2 = x2 * (0 - f1) * (0 - f3) / denom2
                        term3 = x3 * (0 - f1) * (0 - f2) / denom3
                        candidate = term1 + term2 + term3
                        
                        # Accept if within bracket and not too close to endpoints
                        lo, hi = min(a, b), max(a, b)
                        margin = (hi - lo) * 1e-6
                        if lo + margin < candidate < hi - margin:
                            new_x = candidate
            
            if new_x is None:
                # Fallback: bisection
                new_x = (a + b) / 2.0
            
            fx_new = safe_f(new_x)
            if abs(fx_new) <= TOL:
                return new_x
            
            # Update bracket
            if fa * fx_new < 0:
                b, fb = new_x, fx_new
            else:
                a, fa = new_x, fx_new
            
            # Update history: keep last 3, avoid duplicates
            history.append((new_x, fx_new))
            # Remove duplicates (same x or same f)
            unique = []
            for pt in history:
                if not any(abs(pt[0] - u[0]) < 1e-15 and abs(pt[1] - u[1]) < 1e-15 for u in unique):
                    unique.append(pt)
            history = unique[-3:]
        
        return a if abs(fa) < abs(fb) else b
    
    # No bracket found: use secant method with adaptive stepping
    # Start with the two best points we have
    # We already have x, fx and at least one more point from probing
    # Find the second best point
    x1, f1 = x, fx
    x2, f2 = None, None
    
    # Collect all evaluated points (we need to track them)
    # Since we didn't keep a full list, let's re-probe minimally
    # Actually, let's just use secant from x with a small step
    step = 1e-4 * max(1.0, abs(x))
    x2 = x + step
    f2 = safe_f(x2)
    if abs(f2) <= TOL:
        return x2
    
    for _ in range(MAX_EVALS - eval_count):
        if abs(f1) <= TOL:
            return x1
        if abs(f2) <= TOL:
            return x2
        
        # Secant step
        if abs(f2 - f1) > 1e-14:
            new_x = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            # Parallel lines; take a small step
            new_x = x2 + 1e-6 * max(1.0, abs(x2))
        
        fx_new = safe_f(new_x)
        if abs(fx_new) <= TOL:
            return new_x
        
        # Keep the two points with smaller |f|
        if abs(fx_new) < abs(f1):
            x1, f1 = x2, f2
            x2, f2 = new_x, fx_new
        elif abs(fx_new) < abs(f2):
            x2, f2 = new_x, fx_new
        else:
            x1, f1 = x2, f2
            x2, f2 = new_x, fx_new
    
    return x1 if abs(f1) < abs(f2) else x2

# EVOLVE-BLOCK-END
