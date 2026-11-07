"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy: 
    1. Assume f(x) is approximately linear: f(x) ≈ w^T x + b.
    2. Use a small set of deterministic evaluations to estimate w and b via least squares.
       Points: 0, ±e_i * scale.
       This requires 1 + 2*dim = 11 evaluations for dim=5.
    3. Calculate the optimal point x_opt for the linear model within the box [-5, 5]^dim.
       For a linear function, the minimum is at a corner.
       x_i = -5 if w_i > 0, else 5.
    4. Evaluate f(x_opt).
    5. Use remaining budget for deterministic coordinate descent (1D line search) 
       to refine the solution, in case the function has non-linear features 
       or the linear approximation was slightly off due to the specific instance transformation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        def timed_out():
            return time.time() > deadline

        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = None
        best_fx = float('inf')
        calls = 0

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        # --- Phase 1: Linear Model Estimation ---
        # We need to estimate f(x) = w^T x + b.
        # Let's use points: 0, h*e_i, -h*e_i.
        # f(0) = b
        # f(h*e_i) = h*w_i + b
        # f(-h*e_i) = -h*w_i + b
        # w_i = (f(h*e_i) - f(-h*e_i)) / (2*h)
        
        # Choose h. Since domain is [-5, 5], h=1.0 is a good scale.
        # It stays well within bounds from 0.
        h = 1.0
        
        # Evaluate at 0
        x0 = np.zeros(dim)
        fx0 = func(x0)
        calls += 1
        update_best(x0, fx0)
        
        # Check time
        if timed_out():
            return best_x, best_fx

        # Evaluate at +/- h*e_i
        w = np.zeros(dim)
        for i in range(dim):
            if timed_out():
                break
                
            x_plus = np.zeros(dim)
            x_plus[i] = h
            x_minus = np.zeros(dim)
            x_minus[i] = -h
            
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            calls += 2
            update_best(x_plus, f_plus)
            update_best(x_minus, f_minus)
            
            w[i] = (f_plus - f_minus) / (2 * h)
            
            if timed_out():
                break

        # --- Phase 2: Linear Optimum Jump ---
        # For linear f(x) = w^T x + b, min on box is at corners.
        x_linear_opt = np.zeros(dim)
        for i in range(dim):
            if w[i] > 0:
                x_linear_opt[i] = -5.0
            else:
                x_linear_opt[i] = 5.0
                
        f_linear_opt = func(x_linear_opt)
        calls += 1
        update_best(x_linear_opt, f_linear_opt)
        
        if timed_out():
            return best_x, best_fx

        # --- Phase 3: Refinement via Coordinate Descent ---
        # The function might not be perfectly linear, or the "minimizer" constraint 
        # might imply a local structure that the global linear fit misses slightly.
        # Perform deterministic coordinate descent starting from the current best.
        # We alternate between large and small steps to ensure convergence.
        
        x_curr = best_x.copy()
        f_curr = best_fx
        
        # Simple coordinate descent
        # We have budget - calls remaining.
        # Each pass over coordinates costs dim calls.
        # We can do several passes.
        
        # Adaptive step size for coordinate descent
        # Start with a moderate step, shrink if no improvement.
        step_sizes = [2.0, 1.0, 0.5, 0.1, 0.01, 0.001]
        
        for step in step_sizes:
            if timed_out():
                break
            if calls >= budget:
                break
                
            improved = False
            for i in range(dim):
                if timed_out():
                    break
                if calls >= budget:
                    break
                    
                # Try moving in + and - direction along axis i
                # We need to check bounds.
                
                # + direction
                x_try = x_curr.copy()
                x_try[i] += step
                x_try = np.clip(x_try, -5.0, 5.0)
                
                # Avoid redundant evaluation if clipped to same point
                if not np.allclose(x_try, x_curr):
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_curr:
                        x_curr = x_try
                        f_curr = f_try
                        update_best(x_curr, f_curr)
                        improved = True
                    # Check time/budget
                    if timed_out() or calls >= budget:
                        break

                # - direction
                if timed_out() or calls >= budget:
                    break
                    
                x_try = x_curr.copy()
                x_try[i] -= step
                x_try = np.clip(x_try, -5.0, 5.0)
                
                if not np.allclose(x_try, x_curr):
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_curr:
                        x_curr = x_try
                        f_curr = f_try
                        update_best(x_curr, f_curr)
                        improved = True
            
            if not improved:
                # If no improvement with this step size, try smaller one
                continue

        return best_x, best_fx
# EVOLVE-BLOCK-END
