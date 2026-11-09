"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Paraboloid Fit Optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def paraboloid_fit(x0, step_size):
            """
            Fits a paraboloid to 2n+1 points and returns the predicted minimum.
            Requires 2n+1 evaluations.
            """
            nonlocal calls_made, total_budget, deadline

            if calls_made + 2 * n + 1 > total_budget:
                return x0, evaluate(x0)
            if time.time() > deadline:
                return x0, evaluate(x0)

            # Evaluate center
            f0 = evaluate(x0)
            
            # Evaluate points along axes
            f_plus = np.zeros(n)
            f_minus = np.zeros(n)
            x_plus = np.zeros((n, n))
            x_minus = np.zeros((n, n))
            
            for i in range(n):
                x_p = x0.copy()
                x_p[i] += step_size
                x_p = np.clip(x_p, -5, 5)
                x_plus[i] = x_p
                f_plus[i] = evaluate(x_p)
                
                if calls_made >= total_budget or time.time() > deadline:
                    return x0, f0

                x_m = x0.copy()
                x_m[i] -= step_size
                x_m = np.clip(x_m, -5, 5)
                x_minus[i] = x_m
                f_minus[i] = evaluate(x_m)
                
                if calls_made >= total_budget or time.time() > deadline:
                    return x0, f0

            # Solve for the minimum of the quadratic approximation
            # f(x) ~ f0 + sum_i ( (f_plus[i] - f_minus[i]) / (2*step) ) * (x_i - x0_i) + 
            #       sum_i ( (f_plus[i] + f_minus[i] - 2*f0) / (2*step^2) ) * (x_i - x0_i)^2
            # This is a separable quadratic. The minimum for each dimension i is:
            # x_i_min = x0_i - a_i / (2 * b_i)
            # where a_i is the linear coefficient and b_i is the quadratic coefficient.
            
            # a_i = (f_plus[i] - f_minus[i]) / (2 * step)
            # b_i = (f_plus[i] + f_minus[i] - 2 * f0) / (2 * step^2)
            
            x_new = x0.copy()
            for i in range(n):
                a_i = (f_plus[i] - f_minus[i]) / (2 * step_size)
                b_i = (f_plus[i] + f_minus[i] - 2 * f0) / (2 * step_size**2)
                
                if abs(b_i) > 1e-10:
                    # If b_i > 0, it's a local minimum in this dimension
                    # If b_i < 0, it's a local maximum, so we should move away or keep
                    if b_i > 0:
                        x_new[i] = x0[i] - a_i / (2 * b_i)
                    else:
                        # If concave, move to the boundary of the step or keep
                        # Simple heuristic: move towards the better of plus/minus
                        if f_plus[i] < f_minus[i]:
                            x_new[i] = x0[i] + step_size
                        else:
                            x_new[i] = x0[i] - step_size
                else:
                    # Linear or flat
                    if f_plus[i] < f_minus[i]:
                        x_new[i] = x0[i] + step_size
                    elif f_minus[i] < f_plus[i]:
                        x_new[i] = x0[i] - step_size
                    # else keep x0[i]

            x_new = np.clip(x_new, -5, 5)
            
            # Check if we improved
            f_new = evaluate(x_new)
            
            if f_new < f0:
                return x_new, f_new
            else:
                return x0, f0

        # Main loop
        # We use a variable step size that shrinks as we converge
        # Start with a larger step to explore, then shrink
        
        # Initial exploration
        x_current = np.zeros(n)
        f_current = evaluate(x_current)
        
        step_size = 1.0
        min_step = 1e-4
        max_iter = 100
        
        for _ in range(max_iter):
            if calls_made >= total_budget or time.time() > deadline:
                break
                
            # Perform paraboloid fit
            x_next, f_next = paraboloid_fit(x_current, step_size)
            
            if f_next < f_current:
                x_current = x_next
                f_current = f_next
                # Shrink step size if we made progress
                step_size *= 0.7
            else:
                # No progress, shrink step size more aggressively or restart
                step_size *= 0.5
                if step_size < min_step:
                    break
            
            # If we are stuck, try a random restart near best
            if step_size < 0.1 and calls_made < total_budget * 0.5:
                # Random perturbation
                if best_x is not None:
                    x_current = np.clip(best_x + rng.standard_normal(n) * 1.0, -5, 5)
                    f_current = evaluate(x_current)
                    step_size = 0.5
                else:
                    x_current = rng.uniform(-5, 5, n)
                    f_current = evaluate(x_current)
                    step_size = 1.0

        # Final refinement with very small steps
        if calls_made < total_budget and time.time() < deadline:
            step_size = 0.01
            for _ in range(20):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x_next, f_next = paraboloid_fit(x_current, step_size)
                if f_next < f_current:
                    x_current = x_next
                    f_current = f_next
                    step_size *= 0.5
                else:
                    break

        if best_x is None:
            best_x = x_current
            best_f = f_current

        return best_x, best_f
# EVOLVE-BLOCK-END
