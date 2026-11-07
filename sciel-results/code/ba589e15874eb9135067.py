"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Scalar root finder for smooth polynomial equations.
    Returns x with |f(x)| <= 1e-8.
    """
    deadline = time.time() + 110.0  # 110s safety margin (evaluator kills at 120s)
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = f(best_x)
        if not math.isfinite(best_fx):
            best_fx = float('inf')
    except Exception:
        best_fx = float('inf')
    
    def track(x, fx):
        nonlocal best_x, best_fx
        try:
            afx = abs(fx)
            if math.isfinite(afx) and afx < abs(best_fx):
                best_x = x
                best_fx = fx
        except Exception:
            pass
    
    track(best_x, best_fx)
    
    # Phase 1: High-order hybrid method (Newton + secant + bisection fallback)
    # Use a combination of methods to converge quickly with few function calls
    
    x = float(x0)
    
    # Try to establish a bracket first with a few secant steps
    # Then use Newton when derivative is reliable
    
    x_prev = x
    fx_prev = best_fx
    
    for it in range(40):
        if time.time() > deadline:
            break
        
        # Check if we're good enough
        if abs(best_fx) <= 1e-8:
            return best_x
        
        # Compute f(x) if not already known
        try:
            fx = f(x)
        except Exception:
            fx = float('inf')
        
        track(x, fx)
        
        if not math.isfinite(fx):
            # Try to recover by stepping back
            x = x_prev
            break
        
        if abs(fx) <= 1e-8:
            return x
        
        # Compute derivative using central difference
        h = 1e-5 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            fx_plus = float('inf')
            fx_minus = float('inf')
        
        if math.isfinite(fx_plus) and math.isfinite(fx_minus):
            dfx = (fx_plus - fx_minus) / (2 * h)
        elif math.isfinite(fx_plus):
            dfx = (fx_plus - fx) / h
        elif math.isfinite(fx_minus):
            dfx = (fx - fx_minus) / h
        else:
            dfx = 0.0
        
        # Newton step
        if abs(dfx) > 1e-15:
            x_newton = x - fx / dfx
        else:
            x_newton = None
        
        # Secant step
        if abs(fx_prev - fx) > 1e-15:
            x_secant = x - fx * (x - x_prev) / (fx - fx_prev)
        else:
            x_secant = None
        
        # Choose the better step
        candidates = []
        if x_newton is not None and math.isfinite(x_newton):
            candidates.append(('newton', x_newton))
        if x_secant is not None and math.isfinite(x_secant):
            candidates.append(('secant', x_secant))
        
        if not candidates:
            # Fallback: small perturbation
            x_new = x + 1e-3 * max(1.0, abs(x))
        else:
            # Prefer Newton if derivative is well-conditioned
            # Use the candidate that gives smaller |f| estimate (just pick Newton if available)
            if len(candidates) == 1:
                x_new = candidates[0][1]
            else:
                # Use Newton if derivative magnitude is reasonable
                if abs(dfx) > 1e-10:
                    x_new = x_newton
                else:
                    x_new = x_secant
        
        # Damping: if the step is too large, damp it
        step = abs(x_new - x)
        max_step = 10.0 * max(1.0, abs(x))
        if step > max_step:
            x_new = x + max_step * math.copysign(1.0, x_new - x)
        
        x_prev, fx_prev = x, fx
        x = x_new
    
    # Phase 2: Polish with high-precision Newton if we're close
    if abs(best_fx) <= 1e-4:
        x = best_x
        for it in range(20):
            if time.time() > deadline:
                break
            try:
                fx = f(x)
            except Exception:
                break
            if not math.isfinite(fx):
                break
            if abs(fx) <= 1e-10:
                return x
            h = 1e-8 * max(1.0, abs(x))
            try:
                fx_p = f(x + h)
                fx_m = f(x - h)
            except Exception:
                break
            if not (math.isfinite(fx_p) and math.isfinite(fx_m)):
                break
            dfx = (fx_p - fx_m) / (2 * h)
            if abs(dfx) < 1e-15:
                break
            x = x - fx / dfx
            track(x, fx)
    
    # Return best found
    if math.isfinite(best_x):
        return best_x
    return float(x0)
# EVOLVE-BLOCK-END
