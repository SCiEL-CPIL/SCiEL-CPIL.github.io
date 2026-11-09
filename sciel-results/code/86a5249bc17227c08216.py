"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting from x0 using a derivative-free inverse quadratic
    interpolation strategy with bisection fallback for robustness on ill-conditioned
    functions.
    
    Returns x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    """
    MAX_EVALS = 500
    TOL = 1e-8
    
    # State for tracking evaluations
    eval_count = 0
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise ValueError("Evaluation limit exceeded")
        eval_count += 1
        return f(x)
    
    x = float(x0)
    
    # Initial evaluation
    fx = safe_f(x)
    if abs(fx) <= TOL:
        return x
    
    # We'll maintain a list of (x, f(x)) points for interpolation
    # Start with just the initial point
    points = [(x, fx)]
    
    # Try to establish a bracket or gather more points
    # First, try a small step to get a second point
    step = 1e-4 * max(1.0, abs(x))
    x2 = x + step
    fx2 = safe_f(x2)
    if abs(fx2) <= TOL:
        return x2
    points.append((x2, fx2))
    
    # Check if we have a bracket
    if fx * fx2 < 0:
        # We have a bracket, use inverse quadratic interpolation with bisection fallback
        a, b = (x, x2) if fx * fx2 < 0 else (x2, x)
        fa, fb = (fx, fx2) if fx * fx2 < 0 else (fx2, fx)
        
        # Keep a history of points for IQI
        history = [(a, fa), (b, fb)]
        
        for _ in range(MAX_EVALS - eval_count):
            # Check convergence
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            
            # Try inverse quadratic interpolation
            new_x = None
            
            if len(history) >= 3:
                # Fit inverse quadratic to last 3 points
                x1, f1 = history[-3]
                x2_, f2 = history[-2]
                x3, f3 = history[-1]
                
                # Inverse quadratic interpolation: fit p(f) = x
                # Using Lagrange interpolation on (f_i, x_i)
                denom1 = (f1 - f2) * (f1 - f3)
                denom2 = (f2 - f1) * (f2 - f3)
                denom3 = (f3 - f1) * (f3 - f2)
                
                if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                    # Interpolate at f=0
                    term1 = x1 * (0 - f2) * (0 - f3) / denom1
                    term2 = x2_ * (0 - f1) * (0 - f3) / denom2
                    term3 = x3 * (0 - f1) * (0 - f2) / denom3
                    new_x = term1 + term2 + term3
                    
                    # Check if new_x is within the bracket
                    if min(a, b) < new_x < max(a, b):
                        pass
                    else:
                        new_x = None
            
            if new_x is None:
                # Fallback to bisection
                new_x = (a + b) / 2.0
            
            fx_new = safe_f(new_x)
            if abs(fx_new) <= TOL:
                return new_x
            
            # Update bracket
            if fa * fx_new < 0:
                b, fb = new_x, fx_new
            else:
                a, fa = new_x, fx_new
            
            # Add to history, keep last 3
            history.append((new_x, fx_new))
            if len(history) > 3:
                history.pop(0)
        
        # Return the better of a or b
        return a if abs(fa) < abs(fb) else b
    
    else:
        # No bracket, try to find one or use secant-like method
        # Try expanding step
        step = 1e-4 * max(1.0, abs(x))
        for direction in [1, -1]:
            x_try = x + direction * step
            fx_try = safe_f(x_try)
            if abs(fx_try) <= TOL:
                return x_try
            if fx * fx_try < 0:
                # Found a bracket
                a, b = (x, x_try) if fx * fx_try < 0 else (x_try, x)
                fa, fb = (fx, fx_try) if fx * fx_try < 0 else (fx_try, fx)
                history = [(a, fa), (b, fb)]
                
                for _ in range(MAX_EVALS - eval_count):
                    if abs(fa) <= TOL:
                        return a
                    if abs(fb) <= TOL:
                        return b
                    
                    new_x = None
                    if len(history) >= 3:
                        x1, f1 = history[-3]
                        x2_, f2 = history[-2]
                        x3, f3 = history[-1]
                        
                        denom1 = (f1 - f2) * (f1 - f3)
                        denom2 = (f2 - f1) * (f2 - f3)
                        denom3 = (f3 - f1) * (f3 - f2)
                        
                        if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                            term1 = x1 * (0 - f2) * (0 - f3) / denom1
                            term2 = x2_ * (0 - f1) * (0 - f3) / denom2
                            term3 = x3 * (0 - f1) * (0 - f2) / denom3
                            new_x = term1 + term2 + term3
                            
                            if min(a, b) < new_x < max(a, b):
                                pass
                            else:
                                new_x = None
                    
                    if new_x is None:
                        new_x = (a + b) / 2.0
                    
                    fx_new = safe_f(new_x)
                    if abs(fx_new) <= TOL:
                        return new_x
                    
                    if fa * fx_new < 0:
                        b, fb = new_x, fx_new
                    else:
                        a, fa = new_x, fx_new
                    
                    history.append((new_x, fx_new))
                    if len(history) > 3:
                        history.pop(0)
                
                return a if abs(fa) < abs(fb) else b
        
        # No bracket found, use secant method with fallback
        # Use the two best points we have
        if len(points) >= 2:
            x1, f1 = points[0]
            x2_, f2 = points[1]
            history = [(x1, f1), (x2_, f2)]
            
            for _ in range(MAX_EVALS - eval_count):
                if abs(f1) <= TOL:
                    return x1
                if abs(f2) <= TOL:
                    return x2_
                
                # Secant method
                if f2 != f1:
                    new_x = x2_ - f2 * (x2_ - x1) / (f2 - f1)
                else:
                    new_x = x2_ + 1e-6
                
                fx_new = safe_f(new_x)
                if abs(fx_new) <= TOL:
                    return new_x
                
                # Keep the two points with smaller |f|
                if abs(fx_new) < abs(f1):
                    x1, f1 = x2_, f2
                    x2_, f2 = new_x, fx_new
                else:
                    x1, f1 = x2_, f2
                    x2_, f2 = new_x, fx_new
            
            return x1 if abs(f1) < abs(f2) else x2_
        
        # Only one point, return it
        return x

# EVOLVE-BLOCK-END
