"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    # Deadline: 110s leaves 10s safety margin before the 120s limit.
    deadline = start_time + 110.0
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # Track best point found so far (smallest |f|)
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_fx, best_abs_fx
        if math.isfinite(f_val):
            af = abs(f_val)
            if af < best_abs_fx:
                best_abs_fx = af
                best_x = x_val
                best_fx = f_val
                if af <= 1e-8:
                    return True
        return False

    # 2. Geometric Bracket Search
    # We want to find a, b such that f(a) and f(b) have opposite signs.
    # Expand in both directions simultaneously with a smaller growth factor (1.5)
    # to reduce the "wasted" distance of the bracket.
    
    a = x
    fa = fx
    b = None
    fb = None
    
    # We'll keep two expanding points: one to the left, one to the right.
    # However, the standard approach is to pick a direction.
    # Let's try a "double-sided" expansion but we only need one bracket.
    # Actually, expanding in one direction is simpler and often sufficient.
    # But if the root is behind us, we fail.
    # Let's stick to the robust method: try right, then left.
    # Optimization: Use a smaller initial step and factor 1.5 for finer resolution.
    
    # Try expanding to the right first
    step = 1e-3
    for i in range(80):
        if time.time() > deadline:
            return best_x
        
        b = a + step
        try:
            fb = f(b)
        except Exception:
            step *= 1.5
            continue
        
        if not math.isfinite(fb):
            step *= 1.5
            continue
        
        if update_best(b, fb):
            return b
            
        if fa == 0:
            return a
        if fb == 0:
            return b
        
        if fa * fb < 0:
            # Bracket found!
            break
        
        step *= 1.5
    
    if b is None or not math.isfinite(fb) or fa * fb >= 0:
        # No bracket found to the right. Try to the left.
        step = 1e-3
        b = None
        fb = None
        for i in range(80):
            if time.time() > deadline:
                return best_x
            
            b = a - step
            try:
                fb = f(b)
            except Exception:
                step *= 1.5
                continue
            
            if not math.isfinite(fb):
                step *= 1.5
                continue
            
            if update_best(b, fb):
                return b
                
            if fa == 0:
                return a
            if fb == 0:
                return b
            
            if fa * fb < 0:
                break
            
            step *= 1.5
    
    if b is None or not math.isfinite(fb) or fa * fb >= 0:
        # Still no bracket.
        # As a last resort, we could try a local minimization of |f(x)|,
        # but for smooth polynomials, a bracket usually exists unless the root
        # is a tangent (double root) or we are in a weird region.
        # If we haven't found a bracket, return the best point we've seen.
        return best_x

    # 3. Root Refinement
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # Use a hybrid Secant/Bisection method.
    # Secant is faster (superlinear) but can diverge. Bisection is robust but linear.
    # We will use Bisection to maintain the bracket, and Secant to jump.
    # If Secant step stays within bracket, use it. Otherwise, bisect.
    
    x1 = a
    x2 = b
    f1 = fa
    f2 = fb
    
    # Max iterations to be safe, though convergence should be fast.
    max_iter = 100
    
    for i in range(max_iter):
        if time.time() > deadline:
            return best_x
        
        # Check if we've converged
        if abs(f1) < abs(f2):
            # Swap to keep f2 as the smaller one? No, standard secant uses x1, x2.
            # Let's just check tolerance.
            pass
        
        # Try Secant step
        if f1 != f2:
            x_sec = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            # Degenerate case, bisect
            x_sec = (x1 + x2) / 2.0
            f_sec = f(x_sec)
            update_best(x_sec, f_sec)
            if abs(f_sec) <= 1e-8:
                return x_sec
            if x_sec > x1:
                x2 = x_sec
                f2 = f_sec
            else:
                x1 = x_sec
                f1 = f_sec
            continue

        # Check if x_sec is within the bracket [x1, x2]
        if (x_sec > min(x1, x2)) and (x_sec < max(x1, x2)):
            # Use Secant step
            try:
                f_sec = f(x_sec)
            except Exception:
                # Fallback to bisection
                x_sec = (x1 + x2) / 2.0
                try:
                    f_sec = f(x_sec)
                except Exception:
                    break
                update_best(x_sec, f_sec)
                if abs(f_sec) <= 1e-8:
                    return x_sec
                if x_sec > x1:
                    x2 = x_sec
                    f2 = f_sec
                else:
                    x1 = x_sec
                    f1 = f_sec
                continue
            
            if not math.isfinite(f_sec):
                # Fallback to bisection
                x_sec = (x1 + x2) / 2.0
                try:
                    f_sec = f(x_sec)
                except Exception:
                    break
                update_best(x_sec, f_sec)
                if abs(f_sec) <= 1e-8:
                    return x_sec
                if x_sec > x1:
                    x2 = x_sec
                    f2 = f_sec
                else:
                    x1 = x_sec
                    f1 = f_sec
                continue

            update_best(x_sec, f_sec)
            if abs(f_sec) <= 1e-8:
                return x_sec

            # Update bracket
            if f_sec * f1 < 0:
                x2 = x_sec
                f2 = f_sec
            else:
                x1 = x_sec
                f1 = f_sec
        else:
            # Secant step is outside bracket, use Bisection
            x_mid = (x1 + x2) / 2.0
            try:
                f_mid = f(x_mid)
            except Exception:
                break
            
            if not math.isfinite(f_mid):
                break

            update_best(x_mid, f_mid)
            if abs(f_mid) <= 1e-8:
                return x_mid

            if f_mid * f1 < 0:
                x2 = x_mid
                f2 = f_mid
            else:
                x1 = x_mid
                f1 = f_mid
        
        # Convergence check on interval size
        if abs(x2 - x1) < 1e-12:
            # Pick the point with smaller |f|
            if abs(f1) < abs(f2):
                return x1
            else:
                return x2

    # Final Polish: Central Difference Newton Step
    # We have a good candidate. Let's try to squeeze out more precision.
    # Use the best point found.
    x_best = best_x
    f_best = best_fx
    
    h = 1e-5
    try:
        f_plus = f(x_best + h)
        f_minus = f(x_best - h)
    except Exception:
        return x_best
    
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        df = (f_plus - f_minus) / (2 * h)
        if df != 0:
            x_new = x_best - f_best / df
            try:
                f_new = f(x_new)
            except Exception:
                return x_best
            
            if math.isfinite(f_new) and abs(f_new) < abs(f_best):
                return x_new
    
    return x_best

# EVOLVE-BLOCK-END
