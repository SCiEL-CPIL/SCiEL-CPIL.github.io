"""Derivative-free trust-region root finder for ill-conditioned 1D problems."""
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using derivative-free methods.
    Strategy:
      1. Use a 1D Nelder-Mead (Simplex) method to minimize |f(x)|.
      2. Adaptive scaling of the simplex to handle ill-conditioning.
      3. Robust handling of non-finite values.
      4. Fallback to global search if local minimization fails.
    
    Budget: at most 500 evaluations of f.
    """
    target_tol = 1e-8
    max_evals = 500
    eval_count = [0]
    
    # Penalty for non-finite values
    PENALTY = 1e30

    def eval_f(x):
        eval_count[0] += 1
        try:
            val = f(x)
            if not (isinstance(val, (int, float)) or (hasattr(val, '__float__'))):
                val = float(val)
            if math.isnan(val) or math.isinf(val):
                return PENALTY
            return val
        except Exception:
            return PENALTY

    def obj(x):
        """Objective function: |f(x)| with penalty for non-finite."""
        val = eval_f(x)
        if val >= PENALTY:
            return PENALTY
        return abs(val)

    x = float(x0)
    
    # Initial evaluation
    f0 = eval_f(x)
    if abs(f0) <= target_tol:
        return x

    # Initial simplex setup
    # Choose an initial step size h
    # Heuristic: based on magnitude of x0
    h = 1e-4 * max(1.0, abs(x))
    
    # Ensure h is not too small or too large
    if h < 1e-10: h = 1e-10
    if h > 1e10: h = 1e10

    x1 = x + h
    f1 = eval_f(x1)
    
    # Sort points by objective value
    # Point A (best), Point B (worst)
    if abs(f0) <= abs(f1):
        x_a, f_a = x, f0
        x_b, f_b = x1, f1
    else:
        x_a, f_a = x1, f1
        x_b, f_b = x, f0

    # Nelder-Mead iteration
    # Parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction (outside)
    sigma = 0.5   # Shrink
    
    best_x = x_a
    best_f = abs(f_a)
    
    # We keep a history of the best point found
    global_best_x = x_a
    global_best_f = best_f

    # Iteration loop
    # Limit iterations to avoid infinite loops, but rely on eval_count for hard stop
    max_iter = 400
    
    for i in range(max_iter):
        if eval_count[0] >= max_evals - 10:
            break
            
        if best_f <= target_tol:
            return best_x

        # 1. Reflection
        # Reflect x_b across x_a
        x_r = x_a + alpha * (x_a - x_b)
        f_r = eval_f(x_r)
        
        if f_r < best_f:
            # Reflection improved the best point
            # Check for expansion
            x_e = x_a + gamma * (x_a - x_b)
            f_e = eval_f(x_e)
            
            if f_e < f_r:
                # Expansion is better
                x_b, f_b = x_e, f_e
            else:
                # Reflection is better
                x_b, f_b = x_r, f_r
            
            # Update best
            if abs(f_b) < best_f:
                best_x = x_b
                best_f = abs(f_b)
                if best_f < global_best_f:
                    global_best_x = best_x
                    global_best_f = best_f
                    
        else:
            # Reflection did not improve best point
            # Try Contraction
            x_c = x_a + rho * (x_b - x_a)
            f_c = eval_f(x_c)
            
            if f_c < best_f:
                # Contraction improved best point
                x_b, f_b = x_c, f_c
                if abs(f_b) < best_f:
                    best_x = x_b
                    best_f = abs(f_b)
                    if best_f < global_best_f:
                        global_best_x = best_x
                        global_best_f = best_f
            else:
                # Shrink
                # Shrink both points towards the best point
                # Since x_a is the best, we shrink x_b towards x_a
                x_b = x_a + sigma * (x_b - x_a)
                f_b = eval_f(x_b)
                
                # Update best if x_b became better (unlikely but possible due to noise/flat areas)
                if abs(f_b) < best_f:
                    best_x = x_b
                    best_f = abs(f_b)
                    if best_f < global_best_f:
                        global_best_x = best_x
                        global_best_f = best_f

        # Check termination criteria
        if abs(best_f) <= target_tol:
            return best_x
            
        # Check if simplex has collapsed
        if abs(x_a - x_b) < 1e-14 * max(1.0, abs(x_a)):
            # Simplex collapsed, try to restart with a larger step or break
            # If we have evaluations left, try a larger step from best_x
            if eval_count[0] < max_evals - 20:
                h = max(h * 10.0, 1e-4)
                x1 = best_x + h
                f1 = eval_f(x1)
                if abs(f1) < best_f:
                    x_a, f_a = x1, f1
                    x_b, f_b = best_x, best_f
                else:
                    x_a, f_a = best_x, best_f
                    x_b, f_b = x1, f1
                # Update best
                if abs(f_a) < best_f:
                    best_x = x_a
                    best_f = abs(f_a)
                    if best_f < global_best_f:
                        global_best_x = best_x
                        global_best_f = best_f
            else:
                break

    # If Nelder-Mead didn't find a root, do a final check
    if global_best_f <= target_tol:
        return global_best_x

    # Fallback: Use remaining evaluations for a coarse global search around x0
    # This helps if the local minimum is a false minimum or the root is far away
    remaining = max_evals - eval_count[0]
    if remaining > 20:
        # Golden section search in a wider range
        # Range: [x0 - 100*h_initial, x0 + 100*h_initial] or similar
        # Let's use a logarithmic spacing search
        # We will try to find a sign change first
        # If no sign change, minimize |f(x)| over a grid
        
        # Coarse grid search
        # Log-spaced steps from 1e-6 to 1e6
        steps = []
        current = 1e-6
        while current < 1e6:
            steps.append(current)
            current *= 10.0
        
        for h in steps:
            if eval_count[0] >= max_evals - 5:
                break
            x_left = x0 - h
            x_right = x0 + h
            f_left = eval_f(x_left)
            f_right = eval_f(x_right)
            
            # Check for sign change
            if f_left * f_right < 0:
                # Found bracket, use bisection
                a, b = x_left, x_right
                fa, fb = f_left, f_right
                for _ in range(50):
                    if eval_count[0] >= max_evals - 2:
                        break
                    mid = 0.5 * (a + b)
                    fm = eval_f(mid)
                    if abs(fm) <= target_tol:
                        return mid
                    if fa * fm < 0:
                        b = mid
                        fb = fm
                    else:
                        a = mid
                        fa = fm
                # Return best of a, b
                if abs(fa) < abs(fb):
                    if abs(fa) < global_best_f:
                        return a
                    else:
                        return global_best_x
                else:
                    if abs(fb) < global_best_f:
                        return b
                    else:
                        return global_best_x
            
            # Check if points themselves are roots
            if abs(f_left) <= target_tol:
                return x_left
            if abs(f_right) <= target_tol:
                return x_right
            
            # Update global best
            if abs(f_left) < global_best_f:
                global_best_x = x_left
                global_best_f = abs(f_left)
            if abs(f_right) < global_best_f:
                global_best_x = x_right
                global_best_f = abs(f_right)

    return global_best_x

# EVOLVE-BLOCK-END
