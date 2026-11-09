"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic multi-scale coordinate descent for BBOB f4 (Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        call_count = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility, but random enough to avoid bad corners
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Multi-scale coordinate descent
        # Start with a coarse grid to find a good region, then refine.
        
        # 1. Initial sampling to get a reasonable starting point
        # Instead of just 0, sample a few points to avoid getting stuck in a bad local minimum
        # Rastrigin has many local minima.
        
        # Let's try a few random starts and pick the best one as the seed for coordinate descent.
        # But we need to be careful with budget.
        # 5D space. 
        # Let's do 10 random starts.
        starts = []
        for _ in range(10):
            x = rng.uniform(-4, 4, dim)
            f = wrapped_func(x)
            starts.append((f, x))
        
        if not starts:
            return best_x, best_f
            
        starts.sort(key=lambda t: t[0])
        current_x = starts[0][1].copy()
        current_f = starts[0][0]
        
        # 2. Coordinate Descent with Multi-scale steps
        # Steps: 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001
        # For each step size, try +/- step in each dimension.
        # If improvement, accept.
        
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        for step in step_sizes:
            if call_count >= budget or time.time() > deadline:
                break
                
            improved = True
            while improved and call_count < budget and time.time() < deadline:
                improved = False
                for i in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                        
                    # Try +step
                    new_x = current_x.copy()
                    new_x[i] += step
                    if new_x[i] > 5: new_x[i] = 5
                    if new_x[i] < -5: new_x[i] = -5
                    
                    f_new = wrapped_func(new_x)
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
                        improved = True
                        continue # Move to next dimension immediately? Or restart loop? 
                        # Usually coordinate descent iterates through all dims.
                        # Let's stick to one pass per while loop iteration.
                        
                    # Try -step
                    new_x = current_x.copy()
                    new_x[i] -= step
                    if new_x[i] > 5: new_x[i] = 5
                    if new_x[i] < -5: new_x[i] = -5
                    
                    f_new = wrapped_func(new_x)
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
                        improved = True
                        continue
                        
        # 3. Final Polish with very small steps if budget remains
        # Use a local random search or very fine coordinate descent
        if call_count < budget and time.time() < deadline:
            # Fine-tune with random small perturbations
            # This helps if coordinate descent gets stuck on a ridge
            sigma = 0.01
            while call_count < budget and time.time() < deadline:
                # 5 random steps
                for _ in range(5):
                    if call_count >= budget or time.time() > deadline:
                        break
                    dx = rng.normal(0, sigma, dim)
                    new_x = current_x + dx
                    new_x = np.clip(new_x, -5, 5)
                    f_new = wrapped_func(new_x)
                    if f_new < current_f:
                        current_x = new_x
                        current_f = f_new
                        # Keep sigma same or shrink slightly?
                        # Let's shrink slowly
                        sigma *= 0.95
                    else:
                        # If no improvement, shrink sigma to focus
                        sigma *= 0.9
                        if sigma < 1e-5:
                            break
                if sigma < 1e-5:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
