import numpy as np
import time

class Optimizer:
    """
    Optimizer using Powell's Method with adaptive restarts for BBOB f13 (Sharp Ridge).
    Powell's method is well-suited for functions with narrow ridges/valleys
    as it adapts the search directions to align with the principal axes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin for 330s limit

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        # Sample points to find the basin of attraction.
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        def line_search(x, direction, f_start):
            """
            Perform a ternary search line search along a direction from x.
            Returns the best point and value found on the line.
            """
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x, f_start
            
            # Initial bracket
            # Try t=1.0
            x1 = x + 1.0 * direction
            f1 = safe_call(x1)
            if f1 is None:
                return x, f_start
            
            if f1 < f_start:
                # Minimum is likely between 0 and some larger t
                # Expand to find upper bound
                t_high = 1.0
                f_high = f1
                t_low = 0.0
                f_low = f_start
                
                while f_high < f_low:
                    t_low = t_high
                    f_low = f_high
                    t_high *= 2.0
                    if t_high > 10.0:
                        break
                    x_high = x + t_high * direction
                    f_high = safe_call(x_high)
                    if f_high is None:
                        break
                
                # Ternary search in [t_low/2, t_high] or [0, t_high]
                # We know f(0) > f(t_low) and f(t_low) < f(t_high) is not guaranteed, 
                # but f(t_low) was the best in the expansion.
                # Let's search in [0, t_high]
                a = 0.0
                b = t_high
                fa = f_start
                fb = f_high # This might be worse than f_low, but it's an upper bound on t
                # Actually, we should use the best point found so far as the center?
                # Standard ternary search assumes a single minimum in [a,b].
                # Let's just do a few iterations of ternary search.
                
                for _ in range(5): # 5 iterations is enough for precision
                    if calls >= budget or time.time() > deadline:
                        break
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    x_m1 = x + m1 * direction
                    x_m2 = x + m2 * direction
                    f_m1 = safe_call(x_m1)
                    f_m2 = safe_call(x_m2)
                    if f_m1 is None or f_m2 is None:
                        break
                    
                    if f_m1 < f_m2:
                        b = m2
                        fb = f_m2
                    else:
                        a = m1
                        fa = f_m1
                        
                # Pick the best of the evaluated points
                # We have f_start (t=0), f1 (t=1), f_high (t=t_high), f_m1, f_m2
                # Let's just return the best of the last few?
                # To be safe, evaluate the midpoint of the final interval
                t_mid = (a + b) / 2.0
                x_mid = x + t_mid * direction
                f_mid = safe_call(x_mid)
                if f_mid is None:
                    # Return best known
                    candidates = [(0.0, f_start, x.copy()), (1.0, f1, x1.copy()), (t_high, f_high, (x + t_high*direction).copy())]
                    # Add m1, m2 if available
                    # This is getting complicated. Let's simplify.
                    pass
                else:
                    # Compare f_mid with f_start and f1
                    if f_mid < f_start and f_mid < f1:
                        return x_mid, f_mid
                    elif f1 < f_start:
                        return x1, f1
                    else:
                        return x, f_start
            else:
                # f1 >= f_start
                # Try smaller step
                t = 0.5
                x05 = x + 0.5 * direction
                f05 = safe_call(x05)
                if f05 is None:
                    return x, f_start
                if f05 < f_start:
                    # Search in [0, 1]
                    a = 0.0
                    b = 1.0
                    fa = f_start
                    fb = f1
                    for _ in range(5):
                        if calls >= budget or time.time() > deadline:
                            break
                        m1 = a + (b - a) / 3.0
                        m2 = b - (b - a) / 3.0
                        x_m1 = x + m1 * direction
                        x_m2 = x + m2 * direction
                        f_m1 = safe_call(x_m1)
                        f_m2 = safe_call(x_m2)
                        if f_m1 is None or f_m2 is None:
                            break
                        if f_m1 < f_m2:
                            b = m2
                            fb = f_m2
                        else:
                            a = m1
                            fa = f_m1
                    t_mid = (a + b) / 2.0
                    x_mid = x + t_mid * direction
                    f_mid = safe_call(x_mid)
                    if f_mid is not None and f_mid < f_start:
                        return x_mid, f_mid
                    elif f05 < f_start:
                        return x05, f05
                    else:
                        return x, f_start
                else:
                    return x, f_start

        def powell_search(x0, f0, max_iter=10):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            # Initialize directions
            directions = [np.eye(dim)[i] for i in range(dim)]
            x = x0.copy()
            f = f0
            
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return
                
                x_old = x.copy()
                f_old = f
                delta = 0.0
                max_delta = 0.0
                max_i = 0
                
                for i in range(dim):
                    if calls >= budget or time.time() > deadline:
                        return
                    # Line search along direction i
                    x_new, f_new = line_search(x, directions[i], f)
                    x = x_new
                    f = f_new
                    delta += f_old - f_new # Improvement
                    if f_new < f_old - max_delta:
                        max_delta = f_old - f_new
                        max_i = i
                
                # Update directions
                if delta > 2 * max_delta:
                    # Significant improvement
                    # New direction: x_new - x_old
                    new_dir = x - x_old
                    # Replace the direction with the largest improvement with the new conjugate direction
                    directions[max_i] = new_dir
                    # Perform an additional line search along the new direction
                    if calls >= budget or time.time() > deadline:
                        return
                    x_new, f_new = line_search(x, new_dir, f)
                    x = x_new
                    f = f_new
                else:
                    # Check for convergence
                    if abs(delta) < 1e-8:
                        break

            # Update global best
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Main loop: Run Powell with restarts
        while calls < budget and time.time() < deadline:
            if calls >= budget:
                break

            # Run Powell from current best
            f_before = best_f
            powell_search(best_x, best_f, max_iter=10)

            # If we didn't improve much, try a restart
            if best_f > f_before - 1e-6:
                # Try a local restart first
                if calls < budget - 20:
                    # Perturb best point
                    x_perturb = best_x + rng.normal(0, 0.5, size=dim)
                    f_perturb = safe_call(x_perturb)
                    if f_perturb is not None and f_perturb < best_f:
                        best_x = x_perturb
                        best_f = f_perturb
                    else:
                        # Try a global random restart
                        x_new = rng.uniform(-5.0, 5.0, size=dim)
                        f_new = safe_call(x_new)
                        if f_new is None:
                            break
                        if f_new < best_f:
                            best_x = x_new
                            best_f = f_new
                        else:
                            # If no improvement, break to save budget?
                            # Or continue?
                            pass
            else:
                # We improved, continue with Powell from current best
                pass

            # If we are close to budget, do a final local run
            if budget - calls < 30:
                powell_search(best_x, best_f, max_iter=5)
                break

        return best_x, best_f
