import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    max_calls = 450
    call_count = 0
    
    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)
    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_f = fx

    # Newton's Method with Backtracking Line Search
    # This is generally faster (quadratic convergence) than Secant (superlinear) 
    # or Bisection (linear) for smooth polynomials.
    
    x_curr = x
    fx_curr = fx
    h = 1e-5 * max(1.0, abs(x_curr))
    
    # We allow up to 50 iterations of Newton. 
    # Each iteration typically uses 1-3 calls (f(x), f(x+h), f(x-h) for derivative, then f(x_new)).
    # With backtracking, it might use more, but it's usually efficient.
    for i in range(50):
        if not time_ok():
            break
        
        if abs(fx_curr) <= 1e-8:
            return x_curr

        # Estimate derivative using central difference
        # Adaptive h if x is very small or large
        h = 1e-5 * max(1.0, abs(x_curr))
        
        # If x is 0, h is 1e-5. If x is large, h scales.
        # Central difference: (f(x+h) - f(x-h)) / (2h)
        fx_plus = f_call(x_curr + h)
        if not math.isfinite(fx_plus):
            break
        fx_minus = f_call(x_curr - h)
        if not math.isfinite(fx_minus):
            break
            
        if abs(fx_plus) < abs(best_f):
            best_x = x_curr + h
            best_f = fx_plus
        if abs(fx_minus) < abs(best_f):
            best_x = x_curr - h
            best_f = fx_minus

        d = (fx_plus - fx_minus) / (2.0 * h)
        
        if d == 0 or not math.isfinite(d):
            # Derivative is zero or invalid. Try a different step or break.
            # If derivative is zero, we are likely at a local extremum.
            # We can try to move away slightly.
            x_curr += 0.1 * max(1.0, abs(x_curr))
            fx_curr = f_call(x_curr)
            if not math.isfinite(fx_curr):
                break
            continue

        # Newton step
        step = fx_curr / d
        x_new = x_curr - step
        
        # Backtracking Line Search:
        # We want |f(x_new)| < |f(x_curr)|.
        # If not, reduce the step size.
        lambda_factor = 1.0
        fx_new = None
        for _ in range(20):
            if not time_ok():
                break
            if not math.isfinite(x_new):
                break
            
            fx_new = f_call(x_new)
            if not math.isfinite(fx_new):
                lambda_factor *= 0.5
                x_new = x_curr - lambda_factor * step
                continue

            if abs(fx_new) < abs(best_f):
                best_x = x_new
                best_f = fx_new

            # Check if we have a sufficient decrease
            # Simple condition: |f(x_new)| < |f(x_curr)|
            if abs(fx_new) < abs(fx_curr):
                break
            
            # If not, backtrack
            lambda_factor *= 0.5
            x_new = x_curr - lambda_factor * step
            
            # If step becomes too small, break
            if lambda_factor < 1e-10:
                break
        
        if not math.isfinite(x_new) or fx_new is None:
            break

        if abs(fx_new) <= 1e-8:
            return x_new

        # Update current point
        x_curr = x_new
        fx_curr = fx_new

        # Convergence check on x
        if abs(x_new - x_curr) < 1e-12 * max(1.0, abs(x_curr)):
             # We are converging in x, but maybe f is not small enough?
             # If we are stuck, break and return best.
             pass

    return best_x
