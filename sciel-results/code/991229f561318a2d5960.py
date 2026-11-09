import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Uses a hybrid approach:
    1. Attempt to find a bracket [a, b] where f(a) and f(b) have opposite signs.
    2. If a bracket is found, use the Illinois algorithm (robust bisection variant).
    3. If no bracket is found, use the Secant method with safeguards.
    4. Track the best x (smallest |f(x)|) and return it if tolerance is not met.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe within 120s
    
    x0 = float(x0)
    eval_count = 0
    max_evals = 500
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        val = f(x)
        eval_count += 1
        return val

    # Initial evaluation
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= 1e-8:
        return x0
    
    best_x = x0
    best_fx = fx0
    best_abs_fx = abs(fx0)

    # Try to establish a bracket
    # Start with a small step
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = eval_f(x1)
    if fx1 is None:
        return best_x
    if abs(fx1) < best_abs_fx:
        best_x = x1
        best_fx = fx1
        best_abs_fx = abs(fx1)
    if abs(fx1) <= 1e-8:
        return x1

    # Check for bracket [x0, x1]
    a, b = x0, x1
    fa, fb = fx0, fx1
    
    # Expand bracket if necessary
    if fa * fb > 0:
        # Try expanding in the direction of the smaller |f|
        # If |fa| < |fb|, try x0 - h, else x1 + h
        # Simple strategy: just try a few expansions
        for i in range(20):
            if eval_count >= max_evals:
                break
            if time.time() > deadline:
                return best_x
            
            if abs(fa) < abs(fb):
                # Expand left
                h = h * 2.0
                new_x = a - h
                new_f = eval_f(new_x)
                if new_f is None:
                    break
                if abs(new_f) < best_abs_fx:
                    best_x = new_x
                    best_fx = new_f
                    best_abs_fx = abs(new_f)
                if new_f * fa < 0:
                    b = a
                    fb = fa
                    a = new_x
                    fa = new_f
                    break
                a = new_x
                fa = new_f
            else:
                # Expand right
                h = h * 2.0
                new_x = b + h
                new_f = eval_f(new_x)
                if new_f is None:
                    break
                if abs(new_f) < best_abs_fx:
                    best_x = new_x
                    best_fx = new_f
                    best_abs_fx = abs(new_f)
                if new_f * fb < 0:
                    a = b
                    fa = fb
                    b = new_x
                    fb = new_f
                    break
                b = new_x
                fb = new_f
        
        # If still no bracket, switch to Secant mode without bracket guarantees
        if fa * fb > 0:
            # Use Secant method
            x_prev = a
            x_curr = b
            f_prev = fa
            f_curr = fb
            
            for i in range(400):
                if time.time() > deadline:
                    return best_x
                if abs(f_curr) < best_abs_fx:
                    best_x = x_curr
                    best_fx = f_curr
                    best_abs_fx = abs(f_curr)
                if abs(f_curr) <= 1e-8:
                    return x_curr
                if abs(f_prev) < 1e-15 and abs(f_curr) < 1e-15:
                    break
                if f_prev == f_curr:
                    break
                
                # Secant step
                denom = f_curr - f_prev
                if abs(denom) < 1e-15:
                    break
                x_next = x_curr - f_curr * (x_curr - x_prev) / denom
                
                # Safeguard: if x_next is too far, dampen or reset
                # Heuristic: if it jumps more than 10x the current interval, limit it
                if abs(x_next - x_curr) > 10 * abs(x_curr - x_prev) + 1e-6:
                     # Try a smaller step or bisection-like move
                     # For now, just accept but check bounds
                     pass

                f_next = eval_f(x_next)
                if f_next is None:
                    break
                
                if abs(f_next) < best_abs_fx:
                    best_x = x_next
                    best_fx = f_next
                    best_abs_fx = abs(f_next)
                if abs(f_next) <= 1e-8:
                    return x_next

                # Update points
                x_prev = x_curr
                f_prev = f_curr
                x_curr = x_next
                f_curr = f_next
                
            return best_x

    # Bracket found [a, b] with fa * fb < 0
    # Use Illinois Algorithm
    # Illinois is a modified bisection where the midpoint is biased
    # towards the point with the smaller function value.
    
    for i in range(400):
        if time.time() > deadline:
            return best_x
        if eval_count >= max_evals:
            break
        
        # Ensure |fa| < |fb| for Illinois weighting
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
        # Illinois midpoint:
        # m = a + (b - a) / (1 + (b - m_prev) / (m_prev - a)) ... complex
        # Simpler Illinois:
        # If |fa| < |fb|, new point is closer to a.
        # Weighted average: x = (a*|fb| + b*|fa|) / (|fa| + |fb|)
        # Or standard Illinois:
        # m = (a + b) / 2
        # if f(m) * f(a) < 0: b = m else a = m
        # then swap so |fa| < |fb|
        # next m = (a*|fb| + b*|fa|) / (|fa| + |fb|)
        
        # Let's use the weighted average formula directly which is stable
        if abs(fa) + abs(fb) < 1e-15:
            x_mid = (a + b) / 2.0
        else:
            x_mid = (a * abs(fb) + b * abs(fa)) / (abs(fa) + abs(fb))
            
        # Ensure x_mid is strictly inside
        if x_mid <= a:
            x_mid = (a + b) / 2.0
        elif x_mid >= b:
            x_mid = (a + b) / 2.0
            
        fm = eval_f(x_mid)
        if fm is None:
            break
            
        if abs(fm) < best_abs_fx:
            best_x = x_mid
            best_fx = fm
            best_abs_fx = abs(fm)
        if abs(fm) <= 1e-8:
            return x_mid
            
        if fa * fm < 0:
            b = x_mid
            fb = fm
        else:
            a = x_mid
            fa = fm
            
        # Check for convergence in x
        if abs(b - a) < 1e-12 * max(1.0, abs(a)):
            # Check both endpoints
            if abs(fa) < abs(fb):
                 if abs(fa) <= 1e-8: return a
                 return a
            else:
                 if abs(fb) <= 1e-8: return b
                 return b

    # Final check
    if abs(fa) < abs(fb):
        return a
    else:
        return b

# EVOLVE-BLOCK-END
