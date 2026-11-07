import math

def find_root(f, x0):
    """
    Finds a root of f starting from x0.
    Uses Newton-Raphson with damping, falling back to Bisection if needed.
    Returns x such that |f(x)| <= 1e-8.
    """
    x = float(x0)
    
    # Initial evaluation
    try:
        fx = f(x)
    except Exception:
        return x
    
    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Newton-Raphson with Damping ---
    h_base = 1e-5
    max_newton_iter = 20
    x_newton = x
    fx_newton = fx
    
    for i in range(max_newton_iter):
        # Estimate derivative
        h = h_base * max(1.0, abs(x_newton))
        try:
            fx_plus = f(x_newton + h)
            fx_minus = f(x_newton - h)
            if math.isfinite(fx_plus) and math.isfinite(fx_minus):
                df = (fx_plus - fx_minus) / (2 * h)
            else:
                # Fallback to forward difference
                fx_plus = f(x_newton + h)
                if not math.isfinite(fx_plus):
                    break
                df = (fx_plus - fx_newton) / h
        except Exception:
            break
        
        if not math.isfinite(df) or abs(df) < 1e-12:
            break

        # Newton step
        step = fx_newton / df
        x_candidate = x_newton - step
        
        # Damping
        damping = 1.0
        success = False
        for _ in range(10):
            if not math.isfinite(x_candidate):
                damping *= 0.5
                x_candidate = x_newton - damping * step
                continue
            
            try:
                fx_candidate = f(x_candidate)
            except Exception:
                damping *= 0.5
                x_candidate = x_newton - damping * step
                continue
                
            if not math.isfinite(fx_candidate):
                damping *= 0.5
                x_candidate = x_newton - damping * step
                continue
            
            if abs(fx_candidate) < abs(fx_newton):
                success = True
                break
            else:
                damping *= 0.5
                x_candidate = x_newton - damping * step
                
        if not success:
            break
            
        x_newton = x_candidate
        fx_newton = fx_candidate
        
        if abs(fx_newton) <= 1e-8:
            return x_newton
            
        # Stagnation
        if abs(step) < 1e-12 * max(1.0, abs(x_newton)):
            break

    # If Newton succeeded in getting close but not to tolerance, or failed,
    # check if we have a good candidate
    if abs(fx_newton) <= 1e-8:
        return x_newton

    # --- Phase 2: Bisection Fallback ---
    # Try to find a bracket around x_newton (or x0 if Newton failed early)
    center = x_newton if abs(fx_newton) < abs(fx) else x
    f_center = fx_newton if abs(fx_newton) < abs(fx) else fx
    
    if not math.isfinite(f_center):
        return center

    # Expand to find a bracket
    step = 1.0
    a = None
    b = None
    fa = None
    fb = None
    
    for _ in range(50):
        # Try center + step
        x_try = center + step
        try:
            f_try = f(x_try)
            if math.isfinite(f_try):
                if fa is None or fa * f_try > 0:
                    # Same sign as current best or undefined
                    pass
                else:
                    # Sign change between center and x_try
                    a, b = center, x_try
                    fa, fb = f_center, f_try
                    break
        except Exception:
            pass
            
        # Try center - step
        x_try = center - step
        try:
            f_try = f(x_try)
            if math.isfinite(f_try):
                if fa is None or fa * f_try > 0:
                    pass
                else:
                    # Sign change between x_try and center
                    a, b = x_try, center
                    fa, fb = f_try, f_center
                    break
        except Exception:
            pass
            
        step *= 2.0
        
    if a is not None and b is not None:
        # Bisection
        for _ in range(100):
            m = (a + b) / 2.0
            try:
                fm = f(m)
            except Exception:
                break
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            if abs(b - a) < 1e-12:
                break
        return (a + b) / 2.0

    # If no bracket found, return best estimate
    return center if abs(f_center) < abs(fx) else x
