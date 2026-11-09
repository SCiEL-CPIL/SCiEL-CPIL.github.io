"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Stochastic (1+lambda)-ES with adaptive step size and restarts for BBOB f4."""

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
        rng = np.random.default_rng(12345)

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

        # (1+lambda)-ES parameters
        lambda_ = 20  # Offspring count
        # Initial step size: roughly 1/4 of the box width (10/4 = 2.5)
        sigma = 2.5 
        
        # Initialize at origin
        x = np.zeros(dim)
        f_x = wrapped_func(x)
        
        # Track stagnation for restarts
        last_improvement = 0
        stagnation_count = 0
        max_stagnation = 50  # Number of iterations without improvement before restart
        
        # Main loop
        while call_count < budget - 5 and time.time() < deadline - 5:
            # Generate offspring
            best_child_x = x.copy()
            best_child_f = f_x
            
            for i in range(lambda_):
                if call_count >= budget - 5 or time.time() > deadline - 5:
                    break
                # Gaussian perturbation
                z = rng.standard_normal(dim)
                y = x + sigma * z
                f_y = wrapped_func(y)
                
                if f_y < best_child_f:
                    best_child_f = f_y
                    best_child_x = y.copy()
            
            # Update best
            if best_child_f < f_x:
                x = best_child_x
                f_x = best_child_f
                last_improvement = call_count
                stagnation_count = 0
                # Adaptive step size: increase slightly on improvement
                # Cap sigma to avoid huge jumps
                sigma = min(sigma * 1.05, 5.0) 
            else:
                stagnation_count += 1
                # Adaptive step size: decrease on stagnation
                sigma *= 0.9
                
            # Check for restart condition
            if stagnation_count > max_stagnation:
                # Restart: pick a random point in the box
                # This helps escape local minima in Rastrigin-like functions
                x = rng.uniform(-5, 5, size=dim)
                f_x = wrapped_func(x)
                sigma = 2.5  # Reset step size
                stagnation_count = 0
                last_improvement = call_count

        # Final polish: If we have budget left, do a few more random perturbations around best_x
        # Or just ensure we return the best found
        return best_x, best_f

# EVOLVE-BLOCK-END
