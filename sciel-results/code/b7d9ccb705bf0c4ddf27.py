import numpy as np

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses a Trust Region Newton method with finite difference derivatives.
    This approach avoids the requirement for a sign change bracket, making it 
    suitable for cases where the function does not cross zero or where 
    bracketing is difficult due to ill-conditioning.
    """
    x = float(x0)
    
    # Initial evaluation
    try:
        fx = f(x)
        if np.isnan(fx) or np.isinf(fx):
            return x
        if abs(fx) <= 1e-8:
            return x
    except Exception:
        return x

    best_x = x
    best_f = abs(fx)
    
    # Trust region parameters
    delta = 1.0 
    min_delta = 1e-12
    max_delta = 1e6
    
    # Maximum number of function evaluations
    max_evals = 500
    eval_count = 1  # fx already evaluated
    
    def get_h(x_val):
        # Adaptive step size for finite difference
        return 1e-6 * max(1.0, abs(x_val))

    for i in range(max_evals - 1):
        if abs(fx) <= 1e-8:
            return x
            
        # Estimate derivative using central difference
        h = get_h(x)
        dfdx = None
        try:
            x_plus = x + h
            x_minus = x - h
            
            fx_plus = f(x_plus)
            eval_count += 1
            fx_minus = f(x_minus)
            eval_count += 1
            
            if not (np.isnan(fx_plus) or np.isinf(fx_plus)) and not (np.isnan(fx_minus) or np.isinf(fx_minus)):
                dfdx = (fx_plus - fx_minus) / (2.0 * h)
            elif not (np.isnan(fx_plus) or np.isinf(fx_plus)):
                # Forward difference
                dfdx = (fx_plus - fx) / h
            elif not (np.isnan(fx_minus) or np.isinf(fx_minus)):
                # Backward difference
                dfdx = (fx - fx_minus) / h
            else:
                # Both sides failed, try larger h
                h *= 10.0
                x_plus = x + h
                fx_plus = f(x_plus)
                eval_count += 1
                if not (np.isnan(fx_plus) or np.isinf(fx_plus)):
                    dfdx = (fx_plus - fx) / h
                else:
                    break
        except Exception:
            break

        if dfdx is None or np.isnan(dfdx) or np.isinf(dfdx):
            break
            
        if abs(dfdx) < 1e-15:
            # Derivative is effectively zero. 
            # Newton step would be huge. 
            # We can try a small random step or just break.
            # For robustness, let's try to move a small amount in a random direction 
            # or just shrink delta and break if we can't make progress.
            # Actually, if f' is 0, we are at a local extremum of f. 
            # If f(x) != 0, we are stuck. 
            # Return best.
            break

        # Newton step
        step = -fx / dfdx
        
        # Limit step to trust region
        if abs(step) > delta:
            step = delta * np.sign(step)
            
        x_new = x + step
        
        # Ensure x_new is finite
        if not np.isfinite(x_new):
            delta *= 0.5
            if delta < min_delta:
                break
            continue

        try:
            fx_new = f(x_new)
            eval_count += 1
            if np.isnan(fx_new) or np.isinf(fx_new):
                # Bad step, shrink trust region
                delta *= 0.5
                if delta < min_delta:
                    break
                continue
        except Exception:
            delta *= 0.5
            if delta < min_delta:
                break
            continue
        
        # Update best
        if abs(fx_new) < best_f:
            best_x = x_new
            best_f = abs(fx_new)
            
        if abs(fx_new) <= 1e-8:
            return x_new
            
        # Accept or reject step
        if abs(fx_new) < abs(fx):
            # Accept: move to new point, expand trust region
            x = x_new
            fx = fx_new
            delta = min(delta * 2.0, max_delta)
        else:
            # Reject: shrink trust region
            delta *= 0.5
            if delta < min_delta:
                break

    return best_x
