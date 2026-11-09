"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust hybrid approach:
    1. Secant method with adaptive step sizes
    2. Bisection fallback when sign change detected
    3. Newton-like steps when derivative is well-behaved
    4. Deadline guard to ensure we return before 150s wall-clock limit
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin of 10s before 150s limit
    
    x0 = float(x0)
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= 500:
            return None
        val = f(x)
        eval_count += 1
        return val
    
    # Track best result found so far
    best_x = x0
    best_fx = eval_f(x0)
    if best_fx is None:
        return x0
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Two-point secant/bisection state
    x1 = x0
    fx1 = best_fx
    
    # Try multiple strategies
    # Strategy 1: Secant method with adaptive h
    h = 1e-6 * max(1.0, abs(x0))
    x2 = x0 + h
    fx2 = eval_f(x2)
    if fx2 is None:
        return best_x
    
    if abs(fx2) < best_abs:
        best_x = x2
        best_abs = abs(fx2)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Main loop: hybrid secant/bisection
    x_a, f_a = x1, fx1
    x_b, f_b = x2, fx2
    
    for iteration in range(490):
        if time.time() > deadline:
            break
        if eval_count >= 500:
            break
        
        # Check if we have a sign change for bisection
        if f_a * f_b < 0:
            # Bisection step
            x_mid = 0.5 * (x_a + x_b)
            f_mid = eval_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) < best_abs:
                best_x = x_mid
                best_abs = abs(f_mid)
            if best_abs <= 1e-8:
                return best_x
            
            if f_a * f_mid < 0:
                x_b, f_b = x_mid, f_mid
            else:
                x_a, f_a = x_mid, f_mid
        else:
            # Secant step
            if f_b == f_a:
                # Degenerate case, try a small perturbation
                x_new = x_b + 1e-8 * max(1.0, abs(x_b))
                f_new = eval_f(x_new)
                if f_new is None:
                    break
                if abs(f_new) < best_abs:
                    best_x = x_new
                    best_abs = abs(f_new)
                if best_abs <= 1e-8:
                    return best_x
                x_a, f_a = x_b, f_b
                x_b, f_b = x_new, f_new
                continue
            
            x_secant = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            
            # Clamp secant step to avoid huge jumps
            max_step = 10.0 * max(1.0, abs(x_b))
            if abs(x_secant - x_b) > max_step:
                x_secant = x_b + max_step * (1.0 if x_secant > x_b else -1.0)
            
            f_secant = eval_f(x_secant)
            if f_secant is None:
                break
            
            if abs(f_secant) < best_abs:
                best_x = x_secant
                best_abs = abs(f_secant)
            if best_abs <= 1e-8:
                return best_x
            
            # Update points
            if abs(f_secant) < abs(f_a):
                x_a, f_a = x_b, f_b
                x_b, f_b = x_secant, f_secant
            else:
                x_a, f_a = x_secant, f_secant
    
    # Final polish: if we have a bracket, do a few more bisection steps
    if f_a * f_b < 0 and eval_count < 500 and time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline or eval_count >= 500:
                break
            x_mid = 0.5 * (x_a + x_b)
            f_mid = eval_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) < best_abs:
                best_x = x_mid
                best_abs = abs(f_mid)
            if best_abs <= 1e-8:
                return best_x
            if f_a * f_mid < 0:
                x_b, f_b = x_mid, f_mid
            else:
                x_a, f_a = x_mid, f_mid
    
    return best_x
# EVOLVE-BLOCK-END
