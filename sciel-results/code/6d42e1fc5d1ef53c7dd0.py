"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder for ill-conditioned equations using a hybrid
    Secant-Bracket-Brent strategy with damping and global bracketing fallback.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    x = float(x0)
    eval_count = 0
    max_evals = 500

    # Track best solution found
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    eval_count += 1

    def update_best(x_val, fx_val):
        nonlocal best_x, best_fx, best_abs
        ax = abs(fx_val)
        if ax < best_abs:
            best_x = x_val
            best_fx = fx_val
            best_abs = ax

    def check_done():
        nonlocal eval_count
        if best_abs <= 1e-8:
            return True
        if eval_count >= max_evals:
            return True
        if time.time() >= deadline:
            return True
        return False

    # If already good enough
    if check_done():
        return best_x

    # Phase 1: Try to find a bracket [a, b] where f(a)*f(b) < 0
    # Start near x0 and expand exponentially in both directions
    a, b = x, x
    fa, fb = best_fx, best_fx
    step = max(1e-6, abs(x) * 1e-6)
    
    bracket_found = False
    for _ in range(50):
        if check_done():
            break
        # Try expanding in positive direction
        b = x + step
        fb = f(b)
        eval_count += 1
        update_best(b, fb)
        
        if fa * fb < 0:
            a, b = x, b
            fa, fb = best_fx, fb
            bracket_found = True
            break
        
        # Try expanding in negative direction
        a = x - step
        fa = f(a)
        eval_count += 1
        update_best(a, fa)
        
        if fa * fb < 0:
            a, b = a, x
            fa, fb = fa, best_fx
            bracket_found = True
            break
        
        step *= 2.0
        if step > 1e10:
            break

    if check_done():
        return best_x

    if bracket_found:
        # Phase 2: Use Brent's method (or a simplified version) to find root in [a, b]
        # We'll implement a robust bisection + secant hybrid
        # Store previous points for secant method
        x_prev, f_prev = a, fa
        x_curr, f_curr = b, fb
        
        for _ in range(100):
            if check_done():
                break
            
            # Ensure |f_curr| <= |f_prev|
            if abs(f_curr) > abs(f_prev):
                x_prev, x_curr = x_curr, x_prev
                f_prev, f_curr = f_curr, f_prev
            
            # If f_curr is close enough
            if abs(f_curr) <= 1e-8:
                return x_curr
            
            # Bisection fallback: if we're not making progress or secant step is too large
            mid = (a + b) / 2.0
            f_mid = f(mid)
            eval_count += 1
            update_best(mid, f_mid)
            
            if f_mid == 0:
                return mid
            
            # Update bracket
            if f_curr * f_mid < 0:
                a = x_curr
                fa = f_curr
            else:
                b = mid
                fb = f_mid
            
            # Try secant step
            if x_prev != x_curr:
                x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                
                # Check if secant step is within bracket
                if (a < x_secant < b) or (b < x_secant < a):
                    x_next = x_secant
                else:
                    x_next = mid
            else:
                x_next = mid
            
            # Damping: if step is too large, damp it
            if x_prev != x_curr:
                step_size = abs(x_next - x_curr)
                max_step = abs(b - a)
                if step_size > max_step:
                    x_next = x_curr + math.copysign(max_step * 0.5, x_next - x_curr)
            
            f_next = f(x_next)
            eval_count += 1
            update_best(x_next, f_next)
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next
            
            # If bracket is very small, we're done
            if abs(b - a) < 1e-14:
                break

        return best_x
    else:
        # Phase 2 (no bracket): Use damped Newton/secant method
        x_curr = x
        f_curr = best_fx
        x_prev = x + max(1e-8, abs(x) * 1e-8)
        f_prev = f(x_prev)
        eval_count += 1
        update_best(x_prev, f_prev)
        
        for _ in range(200):
            if check_done():
                break
            
            if abs(f_curr) <= 1e-8:
                return x_curr
            
            # Compute derivative estimate
            if x_prev != x_curr:
                d = (f_curr - f_prev) / (x_curr - x_prev)
            else:
                # Finite difference
                h = 1e-6 * max(1.0, abs(x_curr))
                f_plus = f(x_curr + h)
                eval_count += 1
                d = (f_plus - f_curr) / h
            
            if d == 0:
                break
            
            x_newton = x_curr - f_curr / d
            
            # Damping: limit step size
            step_size = abs(x_newton - x_curr)
            max_step = max(1.0, abs(x_curr))
            if step_size > max_step:
                x_newton = x_curr + math.copysign(max_step * 0.5, x_newton - x_curr)
            
            # Also ensure we don't go too far from x0
            if abs(x_newton - x0) > 1e10:
                x_newton = x0 + math.copysign(1e10, x_newton - x0)
            
            f_newton = f(x_newton)
            eval_count += 1
            update_best(x_newton, f_newton)
            
            # Update for secant
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_newton, f_newton
            
            # Convergence check
            if abs(x_newton - x_curr) < 1e-14:
                break

        return best_x

# EVOLVE-BLOCK-END
