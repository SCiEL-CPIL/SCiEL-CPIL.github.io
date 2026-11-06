"""Root finding for ill-conditioned functions using a hybrid Newton-Secant-Bisection approach."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: Newton when derivative is reliable, secant as fallback,
    bisection when a bracket is found, with damping for stability.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    max_evals = 480  # Safety margin for 500 limit
    
    eval_count = 0
    best_x = float(x0)
    best_fabs = float('inf')
    
    def eval_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            return None
        eval_count += 1
        fx = f(x)
        if math.isfinite(fx):
            fabs = abs(fx)
            if fabs < best_fabs:
                best_fabs = fabs
                best_x = x
            return fx
        return None
    
    def time_exceeded():
        return time.time() > deadline
    
    x = float(x0)
    
    # Initial evaluation
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= 1e-8:
        return x
    
    # Initialize secant points
    x_prev = x
    fx_prev = fx
    
    # Try to establish a bracket or get a second point for secant
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is None:
        return best_x
    if abs(fx2) <= 1e-8:
        return x2
    
    # Check if we have a bracket
    a, b = x, x2
    fa, fb = fx, fx2
    if fa * fb < 0:
        # We have a bracket, use bisection with Newton acceleration
        for _ in range(200):
            if time_exceeded() or eval_count >= max_evals:
                break
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            # bisection midpoint
            m = 0.5 * (a + b)
            fm = eval_f(m)
            if fm is None:
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            # Try Newton step from the better point
            # Use secant derivative for Newton-like step
            if abs(b - a) > 1e-15:
                # Secant slope
                slope = (fb - fa) / (b - a)
                if abs(slope) > 1e-30:
                    # Newton-like step from the point with smaller |f|
                    if abs(fa) < abs(fb):
                        x_new = a - fa / slope
                    else:
                        x_new = b - fb / slope
                    # Damp if step is too large
                    if abs(x_new - a) > 0.5 * abs(b - a) or abs(x_new - b) > 0.5 * abs(b - a):
                        x_new = 0.5 * (a + b)
                    # Only accept if it stays in bracket
                    if a < x_new < b or b < x_new < a:
                        fx_new = eval_f(x_new)
                        if fx_new is None:
                            break
                        if abs(fx_new) <= 1e-8:
                            return x_new
                        if fa * fx_new <= 0:
                            b = x_new
                            fb = fx_new
                        else:
                            a = x_new
                            fa = fx_new
    else:
        # No bracket, use secant/Newton hybrid
        x_curr = x2
        fx_curr = fx2
        x_prev = x
        fx_prev = fx
        
        for _ in range(250):
            if time_exceeded() or eval_count >= max_evals:
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            
            # Secant step
            if abs(fx_curr - fx_prev) > 1e-30:
                dx = (fx_curr - fx_prev) / (x_curr - x_prev)
                if abs(dx) > 1e-30:
                    x_new = x_curr - fx_curr / dx
                else:
                    x_new = x_curr
            else:
                # Derivative near zero, use small perturbation
                h = 1e-6 * max(1.0, abs(x_curr))
                x_new = x_curr + h
            
            # Damping: if step is too large, reduce it
            if abs(x_new - x_curr) > 2.0 * max(1.0, abs(x_curr)):
                x_new = x_curr + 0.5 * math.copysign(1.0, x_new - x_curr) * max(1.0, abs(x_curr))
            
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Check if we now have a bracket
            if fx_new * fx_curr < 0:
                # Switch to bisection mode
                a, b = x_curr, x_new
                fa, fb = fx_curr, fx_new
                for _ in range(200):
                    if time_exceeded() or eval_count >= max_evals:
                        break
                    if abs(fa) < abs(fb):
                        a, b = b, a
                        fa, fb = fb, fa
                    m = 0.5 * (a + b)
                    fm = eval_f(m)
                    if fm is None:
                        break
                    if abs(fm) <= 1e-8:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    # Try Newton acceleration
                    if abs(b - a) > 1e-15:
                        slope = (fb - fa) / (b - a)
                        if abs(slope) > 1e-30:
                            if abs(fa) < abs(fb):
                                x_n = a - fa / slope
                            else:
                                x_n = b - fb / slope
                            if abs(x_n - a) > 0.5 * abs(b - a) or abs(x_n - b) > 0.5 * abs(b - a):
                                x_n = 0.5 * (a + b)
                            if a < x_n < b or b < x_n < a:
                                fx_n = eval_f(x_n)
                                if fx_n is None:
                                    break
                                if abs(fx_n) <= 1e-8:
                                    return x_n
                                if fa * fx_n <= 0:
                                    b = x_n
                                    fb = fx_n
                                else:
                                    a = x_n
                                    fa = fx_n
                break
            
            # Update secant points
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_new
            fx_curr = fx_new
    
    return best_x
# EVOLVE-BLOCK-END
