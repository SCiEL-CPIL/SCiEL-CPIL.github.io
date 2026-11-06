import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a hybrid approach: aggressive bracketing, Illinois bisection,
    and Nelder-Mead style local minimization of |f(x)|.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    TOL = 1e-8
    
    eval_count = 0
    best_x = float(x0)
    best_fabs = float('inf')
    
    def eval_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            return None
        if time.time() > deadline:
            return None
        eval_count += 1
        try:
            fx = f(x)
        except Exception:
            return None
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
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Try to find a bracket with multiple step sizes
    h_base = max(1e-3, 1e-3 * abs(x))
    x2 = x + h_base
    fx2 = eval_f(x2)
    if fx2 is None:
        return best_x
    if abs(fx2) <= TOL:
        return x2
    
    bracket_found = False
    if fx * fx2 < 0:
        bracket_found = True
    else:
        # Try larger steps to find a bracket
        for h_factor in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
            if time_exceeded() or eval_count >= max_evals:
                break
            h2 = h_factor * max(1.0, abs(x))
            x3 = x + h2
            fx3 = eval_f(x3)
            if fx3 is None:
                continue
            if abs(fx3) <= TOL:
                return x3
            if fx * fx3 < 0:
                x2, fx2 = x3, fx3
                bracket_found = True
                break
            elif abs(fx3) < abs(fx2):
                x2, fx2 = x3, fx3

    if bracket_found:
        a, b = x, x2
        fa, fb = fx, fx2
        
        # Illinois method (accelerated bisection)
        for _ in range(350):
            if time_exceeded() or eval_count >= max_evals:
                break
            
            # Illinois: damp the side with larger |f|
            if abs(fa) > abs(fb):
                fa = 0.5 * fa
            else:
                fb = 0.5 * fb
            
            # Secant step within bracket
            denom = fb - fa
            if abs(denom) > 1e-30:
                c = (a * fb - b * fa) / denom
            else:
                c = 0.5 * (a + b)
            
            # Ensure c is within [a, b]
            if not (a < c < b):
                c = 0.5 * (a + b)
            
            fc = eval_f(c)
            if fc is None:
                break
            if abs(fc) <= TOL:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            # Check convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
    else:
        # No bracket found. Use Nelder-Mead style minimization of |f(x)|
        # in 1D, which is essentially a golden section search with
        # adaptive step sizes.
        
        # Initialize two points for secant-like search
        x_prev = x
        fx_prev = fx
        x_curr = x2
        fx_curr = fx2
        
        # If we didn't find a bracket, use a local search approach
        # Try secant steps with damping
        for _ in range(200):
            if time_exceeded() or eval_count >= max_evals:
                break
            
            if abs(fx_curr) <= TOL:
                return x_curr
            
            # Compute secant slope
            dx = x_curr - x_prev
            if abs(dx) < 1e-15:
                # Perturb
                x_prev = x_curr - 1e-6
                fx_prev = eval_f(x_prev)
                if fx_prev is None:
                    break
                dx = x_curr - x_prev
                if abs(dx) < 1e-15:
                    break
            
            slope = (fx_curr - fx_prev) / dx
            if abs(slope) < 1e-30:
                # Function is nearly flat, try a small perturbation
                x_new = x_curr + 1e-4 * max(1.0, abs(x_curr))
            else:
                x_new = x_curr - fx_curr / slope
            
            # Line search: ensure we're making progress
            f_curr_abs = abs(fx_curr)
            step = x_new - x_curr
            
            fx_new = None
            for _ls in range(15):
                if time_exceeded() or eval_count >= max_evals:
                    break
                x_trial = x_curr + step
                fx_trial = eval_f(x_trial)
                if fx_trial is None:
                    break
                if abs(fx_trial) < f_curr_abs:
                    fx_new = fx_trial
                    x_new = x_trial
                    break
                # Halve the step
                step *= 0.5
            
            if fx_new is None or time_exceeded():
                break
            
            if abs(fx_new) <= TOL:
                return x_new
            
            # Check if we created a bracket
            if fx_new * fx_curr < 0:
                a, b = x_curr, x_new
                fa, fb = fx_curr, fx_new
                # Switch to Illinois
                for _ in range(350):
                    if time_exceeded() or eval_count >= max_evals:
                        break
                    
                    if abs(fa) > abs(fb):
                        fa = 0.5 * fa
                    else:
                        fb = 0.5 * fb
                    
                    denom = fb - fa
                    if abs(denom) > 1e-30:
                        c = (a * fb - b * fa) / denom
                    else:
                        c = 0.5 * (a + b)
                    
                    if not (a < c < b):
                        c = 0.5 * (a + b)
                    
                    fc = eval_f(c)
                    if fc is None:
                        break
                    if abs(fc) <= TOL:
                        return c
                    
                    if fa * fc < 0:
                        b = c
                        fb = fc
                    else:
                        a = c
                        fa = fc
                    
                    if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                        break
                break
            
            # Update secant points
            x_prev = x_curr
            fx_prev = fx_curr
            x_curr = x_new
            fx_curr = fx_new

    # Final polish: if we're close, try a few more secant/Newton steps
    if best_fabs <= 1e-4 and eval_count < max_evals - 10:
        x_curr = best_x
        fx_curr = eval_f(x_curr)
        if fx_curr is not None and abs(fx_curr) <= TOL:
            return x_curr
        
        # Try secant from a nearby point
        h_polish = 1e-5 * max(1.0, abs(x_curr))
        x_prev = x_curr - h_polish
        fx_prev = eval_f(x_prev)
        if fx_prev is not None:
            for _ in range(40):
                if time_exceeded() or eval_count >= max_evals:
                    break
                if abs(fx_curr) <= TOL:
                    return x_curr
                dx = x_curr - x_prev
                if abs(dx) < 1e-15:
                    break
                slope = (fx_curr - fx_prev) / dx
                if abs(slope) < 1e-30:
                    break
                x_new = x_curr - fx_curr / slope
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx_curr):
                    x_prev, fx_prev = x_curr, fx_curr
                    x_curr, fx_curr = x_new, fx_new
                else:
                    # Damp the step
                    x_new = 0.5 * (x_curr + x_new)
                    fx_new = eval_f(x_new)
                    if fx_new is None:
                        break
                    if abs(fx_new) < abs(fx_curr):
                        x_prev, fx_prev = x_curr, fx_curr
                        x_curr, fx_curr = x_new, fx_new
                    else:
                        break

    return best_x
