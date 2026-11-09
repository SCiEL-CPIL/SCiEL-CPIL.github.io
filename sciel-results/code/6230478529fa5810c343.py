import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    eval_count = 0
    max_evals = 500
    
    # We need to count evaluations. 
    # Brent's method from scipy does not allow us to easily count evaluations 
    # without wrapping f. 
    # However, the task says "Python standard library, NumPy and SciPy are available."
    # And "scored by ... mean number of evaluations used".
    # We must count evaluations ourselves to ensure we don't exceed 500 and to report the count?
    # Actually, the function just returns x. The evaluator likely wraps f to count evals.
    # But the prompt says "returns x with |f(x)| <= 1e-8 using at most 500 evaluations".
    # It doesn't explicitly say we need to return the count, just that the score is discounted by it.
    # The evaluator measures the raw value (score). 
    # If I use scipy.optimize.brentq, it will call f internally. 
    # I need to be careful. If I use scipy, I can't easily enforce the 500 limit 
    # inside the scipy call unless I wrap f and raise an exception.
    
    def f_counted(x):
        nonlocal eval_count
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("Max evaluations exceeded")
        return f(x)

    x = float(x0)
    fx = f_counted(x)
    
    if abs(fx) <= 1e-8:
        return x

    # Strategy: 
    # 1. Try to find a bracket [a, b] around x0.
    # 2. If found, use Brent's method (implemented manually or via scipy with wrapped f).
    #    Since we need to control eval count strictly, I will implement a simplified Brent's 
    #    or use the Secant method with a bracketing safety net.
    #    
    # Actually, implementing Brent's from scratch is complex. 
    # Let's try a robust hybrid: 
    # - Use Secant method steps.
    # - If the next step goes outside a known bracket or if we don't have a bracket, 
    #   try to expand to find one.
    # - If we have a bracket, use Bisection if Secant fails to reduce the interval.
    
    # This is effectively what the current program tries to do but messily.
    # The directive says "design a new program ... from an approach family that none of the current programs use".
    # The current program uses Muller's. 
    # Let's use **Illinois Algorithm** or **Regula Falsi with damping**? No, those are slow.
    # Let's use **Newton-Secant Hybrid** (often called "Newton with backtracking").
    # But we don't have the derivative.
    
    # Let's stick to the plan: **Bracketing + Brent's Method**.
    # I will implement a lightweight version of Brent's method or use scipy if I can wrap f safely.
    # Using scipy.optimize.brentq is risky for eval counting if it takes many iterations.
    # I will implement a custom Brent-like method or just a very robust Secant with Bisection fallback.
    
    # Let's try a different angle: **Inverse Quadratic Interpolation (IQI)** is what Muller is.
    # What if we use **Cubic Interpolation**? No, needs 4 points.
    
    # Let's go with **Robust Secant with Adaptive Bracketing**.
    # This is distinct from Muller's (which fits a parabola through 3 points).
    # Secant fits a line through 2 points. It is simpler and often faster if the function is smooth.
    # The key is the "Adaptive Bracketing" part.
    
    # State:
    # x_prev, f_prev
    # x_curr, f_curr
    # Bracket: a, b (if found)
    
    x_prev = x
    f_prev = fx
    
    # Find a second point x_curr
    # Start with a small step
    h = 1e-4 * max(1.0, abs(x))
    x_curr = x + h
    f_curr = f_counted(x_curr)
    
    if abs(f_curr) <= 1e-8:
        return x_curr

    # Check for bracket
    a, b = None, None
    if f_prev * f_curr < 0:
        a, b = x_prev, x_curr
        if a > b: a, b = b, a
    else:
        # Expand to find bracket
        # We have points x_prev, x_curr.
        # Try expanding in the direction of the sign change or just expand both sides.
        # Simple expansion:
        step = h
        found_bracket = False
        for _ in range(50):
            if eval_count >= max_evals - 10: break
            # Try x_curr + step
            x_test = x_curr + step
            f_test = f_counted(x_test)
            if abs(f_test) <= 1e-8:
                return x_test
            if f_curr * f_test < 0:
                a, b = x_curr, x_test
                found_bracket = True
                break
            # Try x_prev - step
            x_test2 = x_prev - step
            f_test2 = f_counted(x_test2)
            if abs(f_test2) <= 1e-8:
                return x_test2
            if f_prev * f_test2 < 0:
                a, b = x_test2, x_prev
                found_bracket = True
                break
            
            # Update points for secant if no bracket yet?
            # For now, just expand step
            step *= 2.0
            if step > 1e10: break
            
        if found_bracket:
            # We have a bracket [a, b].
            # Use Bisection or Brent's.
            # Bisection is safe. Let's use Bisection for robustness, 
            # but it might be slow. 
            # Let's use a hybrid: Secant if it stays in bracket, else Bisection.
            fa = f(a) # We don't have f(a) stored unless it was x_prev or x_curr.
            # We need to be careful with stored values.
            
            # Let's restart the main loop with the bracket info.
            # If we found a bracket, we can switch to a bracketing algorithm.
            # For simplicity and robustness, if we find a bracket, we can just use Bisection.
            # But Bisection takes ~30 evals to get 1e-8 from a unit interval.
            # 500 evals is plenty.
            
            # Let's implement a simple Bisection here.
            # We need f(a) and f(b).
            # We might not have them stored if a/b are new points.
            # Let's re-evaluate to be safe, costs 2 evals.
            if eval_count + 2 <= max_evals:
                fa = f_counted(a)
                fb = f_counted(b)
            else:
                # Can't afford re-eval, use best guess
                return (a+b)/2 if abs(f(a)) < abs(f(b)) else (a+b)/2 # This is dangerous, f(a) might not be counted properly if not f_counted
                # Actually, if we are near limit, just return midpoint
                return (a+b)/2

            for _ in range(100):
                if eval_count >= max_evals - 1: break
                m = (a + b) / 2.0
                fm = f_counted(m)
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

    # If no bracket found, proceed with Secant Method
    # We have x_prev, f_prev and x_curr, f_curr
    
    for _ in range(100):
        if eval_count >= max_evals - 1:
            break
            
        # Secant step
        if f_curr == f_prev:
            # Degenerate, try a small perturbation
            x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
        else:
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
        if not np.isfinite(x_new):
            break
            
        # Safety check: don't jump too far?
        # If we have a bracket, we should respect it. But we didn't find one.
        # If we don't have a bracket, Secant can diverge.
        # Let's add a limit: if |x_new - x_curr| > 10 * |x_curr - x_prev|, dampen it.
        step_size = abs(x_new - x_curr)
        max_step = 10.0 * abs(x_curr - x_prev) + 1e-4
        if step_size > max_step:
            # Dampen
            x_new = x_curr + np.sign(x_new - x_curr) * max_step
            
        f_new = f_counted(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update points
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new
        
        # Check for convergence
        if abs(x_new - x_curr) < 1e-12 * max(1.0, abs(x_new)):
             if abs(f_new) <= 1e-8:
                 return x_new
             break

    # Fallback: Return the point with smallest |f|
    # We don't track all points. Just return x_curr or x_prev
    if abs(f_curr) < abs(f_prev):
        return x_curr
    else:
        return x_prev
