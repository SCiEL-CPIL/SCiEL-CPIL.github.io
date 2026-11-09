import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill
        
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_func(x):
            nonlocal best_x, best_f, calls_made
            if time.time() > deadline:
                raise TimeoutError
            if calls_made >= total_budget:
                raise BudgetExceeded
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Sampling to find a good starting point
        # Lunacek has a secondary minimum far away. We need to find the global one.
        # 50 random samples in 5D is sparse but better than nothing for initialization.
        # We will use the best of these to start a local search.
        n_samples = min(50, total_budget)
        
        # Try to hit the center and random points
        # The Lunacek function's global minimum is typically around x=0 or shifted.
        # The secondary minimum is at a specific location.
        # Since we don't know the shift, random sampling is the only way.
        
        for i in range(n_samples):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            try:
                safe_func(x)
            except (TimeoutError, BudgetExceeded):
                break

        if best_x is None:
            # Fallback
            return np.zeros(n), 0.0

        # Phase 2: Local Search using a fast, robust method.
        # Nelder-Mead is often faster than CMA-ES in low dimensions when close to the optimum.
        # However, Lunacek is non-convex and has a "trap" (secondary minimum).
        # If we are in the trap, NM might get stuck.
        # Let's use a hybrid: 
        # 1. Run Nelder-Mead for a fixed number of iterations/calls.
        # 2. If progress stalls, try a random restart or a different local method.
        
        # Actually, a simpler and very effective strategy for low-dim black-box with limited budget:
        # Use a (1+1)-ES with adaptive step size, but with periodic restarts if no improvement.
        # Or, use the "Cyclic Coordinate Descent" mentioned in inspiration 1? 
        # Inspiration 1 scored 2.0977 with CCD + kicks.
        # Let's implement a robust version of that or a similar deterministic local search.
        
        # Let's try a structured local search: 
        # Start from best_x.
        # Use a variable step size.
        # Try moving in each coordinate direction (and negative).
        # If no improvement, reduce step size.
        # If step size becomes too small, try a random kick.
        
        current_x = best_x.copy()
        current_f = best_f
        step_size = 0.5  # Initial step
        
        # We have roughly 450 calls left.
        # A CCD cycle in 5D takes 10 evaluations (2 per dim).
        # So we can afford ~45 cycles.
        
        # To handle the non-convexity, we will monitor progress.
        # If we don't improve for, say, 50 calls, we will restart from a new random point 
        # or from the current best with a larger step.
        
        no_improve_count = 0
        last_best_f = current_f
        
        while calls_made < total_budget and time.time() < deadline:
            improved = False
            
            # One cycle of CCD
            for d in range(n):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Try positive direction
                x_plus = current_x.copy()
                x_plus[d] += step_size
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = safe_func(x_plus)
                
                if f_plus < current_f:
                    current_x = x_plus
                    current_f = f_plus
                    improved = True
                    # If we found an improvement, we might want to keep going in this direction?
                    # Standard CCD moves to the best point found in the line search.
                    # For simplicity, let's just take the step if it's better.
                else:
                    # Try negative direction
                    x_minus = current_x.copy()
                    x_minus[d] -= step_size
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = safe_func(x_minus)
                    
                    if f_minus < current_f:
                        current_x = x_minus
                        current_f = f_minus
                        improved = True
                    else:
                        # No improvement in this dimension
                        pass
            
            if improved:
                no_improve_count = 0
                last_best_f = current_f
                # Optionally reduce step size slightly to refine
                step_size *= 0.95
            else:
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.7
                
                if no_improve_count > 3:
                    # Stagnation detected.
                    # Try a random kick to escape local minimum (e.g., the trap).
                    if step_size < 0.01:
                        # If step size is tiny, we are stuck. Restart from a new random point?
                        # Or just increase step size and kick.
                        kick_dir = rng.standard_normal(n)
                        kick_dir /= np.linalg.norm(kick_dir)
                        current_x = current_x + 0.5 * kick_dir
                        current_x = np.clip(current_x, -5, 5)
                        current_f = safe_func(current_x)
                        step_size = 0.5
                        no_improve_count = 0
                    else:
                        # Just reduce step size more aggressively or kick
                        if no_improve_count > 5:
                            kick_dir = rng.standard_normal(n)
                            kick_dir /= np.linalg.norm(kick_dir)
                            current_x = current_x + 0.2 * kick_dir
                            current_x = np.clip(current_x, -5, 5)
                            current_f = safe_func(current_x)
                            step_size = 0.2
                            no_improve_count = 0

        # Update global best if local search found something better
        if current_f < best_f:
            best_f = current_f
            best_x = current_x.copy()

        return best_x, best_f
