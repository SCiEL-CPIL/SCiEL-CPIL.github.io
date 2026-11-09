"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using Steffensen's method with adaptive
    perturbation for quadratic convergence, with bisection fallback and
    geometric expansion for bracketing.
    """
    tol = 1e-8
    deadline = time.time() + 110.0  # safety margin before 120s wall-clock limit
    
    x = float(x0)
    
    # Evaluate at starting point
    try:
        fx = f(x)
    except Exception:
        fx = float('nan')
    
    if not math.isfinite(fx):
        # Try a few nearby points
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 1.0, -1.0]:
            try:
                x_try = x + dx
                fx_try = f(x_try)
                if math.isfinite(fx_try) and abs(fx_try) <= tol:
                    return x_try
            except Exception:
                continue
        return x  # best effort
    
    if abs(fx) <= tol:
        return x
    
    # Phase 1: Find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x
    fa = fx
    
    # Adaptive geometric expansion: try both directions with exponentially growing steps
    b = None
    fb = None
    
    # Try positive direction
    step = 1e-8
    while step <= 1e6:
        if time.time() > deadline:
            break
        x_try = x + step
        try:
            f_try = f(x_try)
        except Exception:
            step *= 10
            continue
        
        if not math.isfinite(f_try):
            step *= 10
            continue
        
        if abs(f_try) <= tol:
            return x_try
        
        if fa * f_try < 0:
            b = x_try
            fb = f_try
            break
        step *= 10
    
    # If no bracket found in positive direction, try negative direction
    if b is None:
        step = 1e-8
        while step <= 1e6:
            if time.time() > deadline:
                break
            x_try = x - step
            try:
                f_try = f(x_try)
            except Exception:
                step *= 10
                continue
            
            if not math.isfinite(f_try):
                step *= 10
                continue
            
            if abs(f_try) <= tol:
                return x_try
            
            if fa * f_try < 0:
                b = x_try
                fb = f_try
                break
            step *= 10
    
    if b is None:
        # No bracket found. Try Steffensen's method directly from x0
        # Steffensen: x_{n+1} = x_n - f(x_n) / (f(x_n + f(x_n)) - f(x_n))
        # But for oscillatory functions, use adaptive perturbation
        best_x = x
        best_abs_fx = abs(fx)
        
        # First, do a quick local search to find a better starting point
        for scale in [1e-4, 1e-2, 1.0, 10.0, 100.0]:
            if time.time() > deadline:
                break
            for sign in [1, -1]:
                x_try = x + sign * scale
                try:
                    f_try = f(x_try)
                except Exception:
                    continue
                if not math.isfinite(f_try):
                    continue
                if abs(f_try) < best_abs_fx:
                    best_abs_fx = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= tol:
                    return x_try
        
        # Use Steffensen from the best point
        x_curr = best_x
        try:
            f_curr = f(x_curr)
        except Exception:
            return best_x
        
        if not math.isfinite(f_curr):
            return best_x
        
        if abs(f_curr) <= tol:
            return x_curr
        
        # Steffensen's method with adaptive perturbation
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Adaptive perturbation: scale with |f(x)| to handle oscillations
            # For small |f|, use small perturbation; for large |f|, use larger
            delta = max(1e-10, min(1.0, abs(f_curr) * 0.1))
            
            x_perturbed = x_curr + delta
            try:
                f_perturbed = f(x_perturbed)
            except Exception:
                break
            
            if not math.isfinite(f_perturbed):
                # Try negative perturbation
                x_perturbed = x_curr - delta
                try:
                    f_perturbed = f(x_perturbed)
                except Exception:
                    break
                if not math.isfinite(f_perturbed):
                    break
            
            denom = f_perturbed - f_curr
            if abs(denom) < 1e-15:
                break
            
            x_new = x_curr - f_curr * delta / denom
            
            if not math.isfinite(x_new):
                break
            
            try:
                f_new = f(x_new)
            except Exception:
                break
            
            if not math.isfinite(f_new):
                break
            
            if abs(f_new) <= tol:
                return x_new
            
            # Check for divergence: if |f_new| > |f_curr| * 10, revert
            if abs(f_new) > abs(f_curr) * 10:
                # Try bisection-like step toward where f was smaller
                if abs(f_new) > abs(f_perturbed):
                    x_new = x_perturbed
                    f_new = f_perturbed
                else:
                    # Step back
                    x_new = (x_curr + x_perturbed) / 2.0
                    try:
                        f_new = f(x_new)
                    except Exception:
                        break
                    if not math.isfinite(f_new):
                        break
            
            x_curr = x_new
            f_curr = f_new
            
            # Convergence check
            if abs(f_curr) < 1e-14:
                break
        
        return x_curr
    
    # Phase 2: We have a bracket [a, b] with f(a)*f(b) < 0
    # Use Steffensen's method with bisection fallback for guaranteed convergence
    x_lo = a
    x_hi = b
    f_lo = fa
    f_hi = fb
    
    # Start Steffensen from the midpoint or the point with smaller |f|
    if abs(f_lo) <= abs(f_hi):
        x_curr = x_lo
        f_curr = f_lo
    else:
        x_curr = x_hi
        f_curr = f_hi
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Adaptive perturbation for Steffensen
        # Scale perturbation with bracket width and |f|
        bracket_width = abs(x_hi - x_lo)
        delta = max(1e-10, min(bracket_width * 0.01, abs(f_curr) * 0.1 + 1e-10))
        
        x_perturbed = x_curr + delta
        # Ensure perturbed point stays within bracket
        if x_perturbed > x_hi:
            x_perturbed = x_hi - delta
        if x_perturbed < x_lo:
            x_perturbed = x_lo + delta
        
        try:
            f_perturbed = f(x_perturbed)
        except Exception:
            # Fall back to bisection
            x_new = (x_lo + x_hi) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= tol:
                return x_new
            if f_lo * f_new < 0:
                x_hi = x_new
                f_hi = f_new
            else:
                x_lo = x_new
                f_lo = f_new
            continue
        
        if not math.isfinite(f_perturbed):
            # Fall back to bisection
            x_new = (x_lo + x_hi) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= tol:
                return x_new
            if f_lo * f_new < 0:
                x_hi = x_new
                f_hi = f_new
            else:
                x_lo = x_new
                f_lo = f_new
            continue
        
        denom = f_perturbed - f_curr
        if abs(denom) < 1e-15:
            # Fall back to bisection
            x_new = (x_lo + x_hi) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= tol:
                return x_new
            if f_lo * f_new < 0:
                x_hi = x_new
                f_hi = f_new
            else:
                x_lo = x_new
                f_lo = f_new
            continue
        
        x_new = x_curr - f_curr * delta / denom
        
        # Check if Steffensen step is within bracket
        if not (x_lo <= x_new <= x_hi):
            # Clamp to bracket or use bisection
            x_new = (x_lo + x_hi) / 2.0
        
        try:
            f_new = f(x_new)
        except Exception:
            x_new = (x_lo + x_hi) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if not math.isfinite(f_new):
            x_new = (x_lo + x_hi) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) <= tol:
            return x_new
        
        # Update bracket
        if f_lo * f_new < 0:
            x_hi = x_new
            f_hi = f_new
        else:
            x_lo = x_new
            f_lo = f_new
        
        # Update current point for next Steffensen iteration
        x_curr = x_new
        f_curr = f_new
        
        # Convergence check
        if abs(x_hi - x_lo) < 1e-15 * max(1.0, abs(x_new)):
            break
    
    # Return the point with smallest |f| among candidates
    candidates = [x_lo, x_hi, x_curr]
    best_x = x_curr
    best_val = abs(f_curr)
    for cx in candidates:
        if cx == x_curr:
            continue
        try:
            cv = abs(f(cx))
            if math.isfinite(cv) and cv < best_val:
                best_val = cv
                best_x = cx
        except Exception:
            continue
    
    return best_x

# EVOLVE-BLOCK-END
