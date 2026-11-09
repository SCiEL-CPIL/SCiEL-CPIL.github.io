import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find a root x of f(x) such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust hybrid strategy:
    1. Initial sampling to find a bracket or a good starting pair.
    2. If a bracket is found, use Brent's method (via scipy) with strict evaluation counting.
    3. If no bracket is found, use a damped Secant method with adaptive step control.
    4. Fallback to Bisection if bracket is found late.
    """
    eval_count = 0
    max_evals = 500
    tol = 1e-8

    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("Max evaluations exceeded")
        try:
            val = f(x)
            if not np.isfinite(val):
                return np.nan
            return val
        except Exception:
            return np.nan

    x = float(x0)
    fx = f_counted(x)
    
    if np.isnan(fx):
        # If x0 is invalid, try a small perturbation
        x = x + 1e-4
        fx = f_counted(x)
        if np.isnan(fx):
            return x # Can't do much

    if abs(fx) <= tol:
        return x

    # Strategy 1: Try to find a bracket around x0
    # We'll do a geometric expansion in both directions.
    # Keep track of the best point (smallest |f|) and any bracket found.
    
    best_x = x
    best_f = fx
    best_abs_f = abs(fx)

    a = None
    b = None
    fa = None
    fb = None

    # Initial step size
    h = 1e-3 * max(1.0, abs(x))
    
    # We have point (x, fx).
    # Let's try to find a sign change.
    # We'll expand in the direction where f is smaller? Or just both sides?
    # Let's try a few points in both directions.
    
    points = [(x, fx)]
    
    # Try expanding to the right
    x_right = x + h
    f_right = f_counted(x_right)
    if not np.isnan(f_right):
        points.append((x_right, f_right))
        if abs(f_right) < best_abs_f:
            best_x = x_right
            best_f = f_right
            best_abs_f = abs(f_right)
        if fx * f_right < 0:
            a, b = x, x_right
            fa, fb = fx, f_right
            if a > b:
                a, b = b, a
                fa, fb = fb, fa

    # Try expanding to the left
    x_left = x - h
    f_left = f_counted(x_left)
    if not np.isnan(f_left):
        points.append((x_left, f_left))
        if abs(f_left) < best_abs_f:
            best_x = x_left
            best_f = f_left
            best_abs_f = abs(f_left)
        if fx * f_left < 0:
            a, b = x_left, x
            fa, fb = f_left, fx
            if a > b:
                a, b = b, a
                fa, fb = fb, fa

    # If no bracket yet, try expanding further
    if a is None:
        step = h
        for _ in range(30):
            if eval_count > max_evals - 20:
                break
            
            # Expand right
            x_new = x + step
            f_new = f_counted(x_new)
            if not np.isnan(f_new):
                if abs(f_new) < best_abs_f:
                    best_x = x_new
                    best_f = f_new
                    best_abs_f = abs(f_new)
                # Check against last right point
                if x_right is not None and f_right * f_new < 0:
                    a, b = x_right, x_new
                    fa, fb = f_right, f_new
                    if a > b:
                        a, b = b, a
                        fa, fb = fb, fa
                    break
                x_right = x_new
                f_right = f_new
            
            # Expand left
            x_new_left = x - step
            f_new_left = f_counted(x_new_left)
            if not np.isnan(f_new_left):
                if abs(f_new_left) < best_abs_f:
                    best_x = x_new_left
                    best_f = f_new_left
                    best_abs_f = abs(f_new_left)
                if x_left is not None and f_left * f_new_left < 0:
                    a, b = x_new_left, x_left
                    fa, fb = f_new_left, f_left
                    if a > b:
                        a, b = b, a
                        fa, fb = fb, fa
                    break
                x_left = x_new_left
                f_left = f_new_left
                
            step *= 2.0
            if step > 1e10:
                break

    # If we found a bracket, use Brent's method
    if a is not None and b is not None:
        # Ensure we have enough evals left
        if eval_count < max_evals - 10:
            try:
                # Use scipy's brentq with our counted function
                # Brent's method is very efficient, usually < 100 evals for 1e-8
                root = brentq(f_counted, a, b, xtol=1e-12, rtol=1e-12, maxiter=500)
                if abs(f_counted(root)) <= tol: # Final check, costs 1 eval
                    return root
                # If it didn't quite make it, return the best we got
                return root
            except Exception:
                pass

        # Fallback to manual bisection if scipy failed or we are low on evals
        # Re-evaluate a and b to be safe? No, we have fa, fb.
        # But we need to be careful if we used up evals.
        for _ in range(100):
            if eval_count >= max_evals - 1:
                break
            m = (a + b) / 2.0
            fm = f_counted(m)
            if np.isnan(fm):
                break
            if abs(fm) <= tol:
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

    # If no bracket found, use Secant Method with damping
    # We need two points. Use x and the best_x if it's different, or x_right/x_left.
    # Let's pick the two points with smallest |f| from our sampled points.
    
    valid_points = [(px, pf) for px, pf in points if not np.isnan(pf)]
    if len(valid_points) < 2:
        # Not enough points, try to generate one more
        x_new = x + 1e-4 * max(1.0, abs(x))
        f_new = f_counted(x_new)
        if not np.isnan(f_new):
            valid_points.append((x_new, f_new))
    
    if len(valid_points) < 2:
        return best_x

    # Sort by |f|
    valid_points.sort(key=lambda p: abs(p[1]))
    x1, f1 = valid_points[0]
    x2, f2 = valid_points[1]
    
    # If they are the same, pick next
    if x1 == x2 and len(valid_points) > 2:
        x2, f2 = valid_points[2]

    # Secant loop
    for _ in range(200):
        if eval_count >= max_evals - 1:
            break
            
        if f2 == f1:
            # Degenerate
            x_new = x2 + 1e-4 * max(1.0, abs(x2))
        else:
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            
        if not np.isfinite(x_new):
            break
            
        # Damping: limit step size
        step_size = abs(x_new - x2)
        max_step = 5.0 * abs(x2 - x1) + 1e-4
        if step_size > max_step:
            x_new = x2 + np.sign(x_new - x2) * max_step
            
        f_new = f_counted(x_new)
        
        if np.isnan(f_new):
            break
            
        if abs(f_new) <= tol:
            return x_new
            
        if abs(f_new) < best_abs_f:
            best_x = x_new
            best_f = f_new
            best_abs_f = abs(f_new)
            
        # Update points: keep the two with smallest |f|?
        # Or just shift? Standard secant shifts.
        # But for robustness, let's keep the best two.
        # Actually, standard secant is faster if it converges.
        # Let's shift: x1, f1 = x2, f2; x2, f2 = x_new, f_new
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        
        # Check for convergence
        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x2)):
            if abs(f2) <= tol:
                return x2
            break

    return best_x
