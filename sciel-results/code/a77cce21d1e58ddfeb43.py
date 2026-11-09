"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    """
    Find a root of f using a robust combination of:
    1. Bracketing and Brent's method (derivative-free, super-linear)
    2. Bounded minimization of f(x)^2 for cases where no bracket is found
    """
    x0 = float(x0)
    eval_count = 0
    max_evals = 500
    target_tol = 1e-8

    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Evaluation limit exceeded")
        eval_count += 1
        return float(f(x))

    def eval_f_sq(x):
        fx = eval_f(x)
        return fx * fx

    # 1. Initial evaluation
    fx0 = eval_f(x0)
    if abs(fx0) <= target_tol:
        return x0

    # 2. Try to find a bracket
    # We expand the interval [x0-h, x0+h] exponentially
    h = 1e-6 * max(1.0, abs(x0))
    a = b = None
    fa = fb = None
    
    # We keep track of the best point found during scanning
    best_x = x0
    best_f_abs = abs(fx0)

    for i in range(100): # Max 100 expansions
        if eval_count >= max_evals - 10: # Leave some for solver
            break
            
        a = x0 - h
        b = x0 + h
        
        try:
            fa = eval_f(a)
        except:
            break
        if abs(fa) < best_f_abs:
            best_f_abs = abs(fa)
            best_x = a
        if abs(fa) <= target_tol:
            return a

        try:
            fb = eval_f(b)
        except:
            break
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b
        if abs(fb) <= target_tol:
            return b

        # Check for bracket
        if fa * fb < 0:
            # Found a bracket [a, b]
            try:
                # Use brentq which is very efficient
                root = brentq(eval_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if abs(eval_f(root)) <= target_tol:
                    return root
                return root
            except Exception:
                pass

        # Expand h
        h *= 2.0
        if h > 1e10:
            break

    # 3. If no bracket found, use bounded minimization of f(x)^2
    # We use the best_x found during scanning as a hint
    
    def objective(x):
        return eval_f_sq(x)

    # Try increasing bounds
    scales = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    base_scale = max(1.0, abs(best_x))
    
    last_best_x = best_x
    last_best_val = best_f_abs

    for scale in scales:
        if eval_count >= max_evals - 5:
            break
            
        R = scale * base_scale
        if R < 1e-12:
            R = 1e-12
            
        bounds = (best_x - R, best_x + R)
        
        # Ensure bounds are valid
        if bounds[0] >= bounds[1]:
            continue

        try:
            res = minimize_scalar(objective, bounds=bounds, method='bounded', 
                                  options={'xatol': 1e-12})
            
            if res.success:
                x_opt = res.x
                f_opt = eval_f(x_opt)
                
                if abs(f_opt) < last_best_val:
                    last_best_val = abs(f_opt)
                    last_best_x = x_opt
                    
                if abs(f_opt) <= target_tol:
                    return x_opt
                    
        except Exception:
            continue

    # If we still haven't found a root, return the best candidate
    return last_best_x
# EVOLVE-BLOCK-END
