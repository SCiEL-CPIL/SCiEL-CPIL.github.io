"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy: Adaptive Bracketing + Secant Hybrid with High-Precision Newton Polish.
    1. Expand a bracket around x0 geometrically until a sign change is found.
    2. Use secant method within the bracket (faster than pure bisection).
    3. If secant diverges or fails, fall back to bisection.
    4. Final Newton polish with very tight tolerance.
    """
    deadline = time.time() + 140.0  # Safety margin
    tol = 1e-10  # Tighter internal tolerance
    x0 = float(x0)
    
    # Global best tracking
    best_x = x0
    try:
        best_f = f(x0)
    except Exception:
        best_f = float('inf')
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= tol:
        return best_x

    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f

    # Step 1: Find a bracket [a, b] where f(a)*f(b) < 0
    # Start with a small interval and expand geometrically
    a = x0 - 1e-3
    b = x0 + 1e-3
    fa = None
    fb = None
    
    try:
        fa = f(a)
        fb = f(b)
    except Exception:
        pass
    
    update_best(a, fa) if fa is not None else None
    update_best(b, fb) if fb is not None else None
    
    if fa is not None and fb is not None:
        if abs(fa) <= tol:
            return a
        if abs(fb) <= tol:
            return b
        if fa * fb < 0:
            # Bracket found
            pass
        else:
            # Expand geometrically
            scale = 1e-3
            max_expansions = 50
            for _ in range(max_expansions):
                if time.time() > deadline:
                    break
                scale *= 2.0
                a = x0 - scale
                b = x0 + scale
                try:
                    fa = f(a)
                    fb = f(b)
                except Exception:
                    break
                update_best(a, fa)
                update_best(b, fb)
                if abs(fa) <= tol:
                    return a
                if abs(fb) <= tol:
                    return b
                if fa * fb < 0:
                    break
    
    # If no bracket found, try secant from x0 with a small offset
    if fa is None or fb is None or fa * fb >= 0:
        # Use secant method with two nearby points
        x_a = x0
        x_b = x0 + 1e-3
        try:
            f_a = f(x_a)
            f_b = f(x_b)
        except Exception:
            return best_x
        update_best(x_a, f_a)
        update_best(x_b, f_b)
        
        if abs(f_a) <= tol:
            return x_a
        if abs(f_b) <= tol:
            return x_b
        
        # Secant iteration
        for _ in range(30):
            if time.time() > deadline:
                break
            denom = f_b - f_a
            if denom == 0 or not math.isfinite(denom):
                break
            x_new = x_b - f_b * (x_b - x_a) / denom
            if not math.isfinite(x_new):
                break
            x_a, f_a = x_b, f_b
            x_b = x_new
            try:
                f_b = f(x_b)
            except Exception:
                break
            update_best(x_b, f_b)
            if abs(f_b) <= tol:
                return x_b
            if abs(f_a) <= tol:
                return x_a
        
        # Newton polish
        x = best_x
        for _ in range(20):
            if time.time() > deadline:
                break
            try:
                fx = f(x)
            except Exception:
                break
            update_best(x, fx)
            if abs(fx) <= tol:
                return x
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx2 = f(x + h)
            except Exception:
                break
            update_best(x + h, fx2)
            d = (fx2 - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            max_step = 5.0 * max(1.0, abs(x))
            step = x_new - x
            if abs(step) > max_step:
                x_new = x + math.copysign(max_step, step)
            x = x_new
        return best_x
    
    # Step 2: Secant method within bracket [a, b]
    x_a, f_a = a, fa
    x_b, f_b = b, fb
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        if abs(f_a) <= tol:
            return x_a
        if abs(f_b) <= tol:
            return x_b
        
        denom = f_b - f_a
        if denom == 0 or not math.isfinite(denom):
            break
        
        x_new = x_b - f_b * (x_b - x_a) / denom
        
        if not math.isfinite(x_new):
            break
        
        # Check if x_new is within bracket
        if (x_new < min(x_a, x_b) - 1e-10) or (x_new > max(x_a, x_b) + 1e-10):
            # Secant diverged, fall back to bisection
            x_mid = (x_a + x_b) / 2.0
            try:
                f_mid = f(x_mid)
            except Exception:
                break
            update_best(x_mid, f_mid)
            if abs(f_mid) <= tol:
                return x_mid
            if f_mid * f_a < 0:
                x_b, f_b = x_mid, f_mid
            else:
                x_a, f_a = x_mid, f_mid
            continue
        
        update_best(x_new, f(x_new) if False else None)  # Don't call f twice
        # Actually, we need f(x_new)
        try:
            f_new = f(x_new)
        except Exception:
            break
        update_best(x_new, f_new)
        
        if abs(f_new) <= tol:
            return x_new
        
        # Shift window
        x_a, f_a = x_b, f_b
        x_b, f_b = x_new, f_new
    
    # Step 3: Newton polish on best point
    x = best_x
    for _ in range(30):
        if time.time() > deadline:
            break
        try:
            fx = f(x)
        except Exception:
            break
        update_best(x, fx)
        if abs(fx) <= tol:
            return x
        
        h = 1e-8 * max(1.0, abs(x))
        try:
            fx2 = f(x + h)
        except Exception:
            break
        update_best(x + h, fx2)
        
        d = (fx2 - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        max_step = 1.0 * max(1.0, abs(x))
        step = x_new - x
        if abs(step) > max_step:
            x_new = x + math.copysign(max_step, step)
        
        x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
