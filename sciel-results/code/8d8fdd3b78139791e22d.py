# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive (1+1)-ES with 1/5 rule, stall detection, and center-biased restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        calls_made = [0]
        best_f = np.inf
        best_x = np.zeros(n)

        def wrapped_func(x):
            nonlocal best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            calls_made[0] += 1
            if calls_made[0] > budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Strategy: Use a single adaptive (1+1)-ES with stall-triggered restarts.
        # This combines the efficiency of the (1+1)-ES (inspiration 1 & 2) with
        # the exploration capability of restarts (inspiration 3) but avoids the
        # overhead and complexity of Nelder-Mead (current program).
        
        # Parameters
        window_size = 5
        successes = 0
        steps_in_window = 0
        stall_threshold = 40  # Steps without improvement before restarting
        max_sigma = 5.0       # Cap on sigma to prevent divergence
        min_sigma = 1e-4      # Floor on sigma
        
        # Initial state
        # Start at center as a strong prior, but we will restart if we get stuck.
        x = np.zeros(n)
        fx = wrapped_func(x)
        sigma = 1.5  # Slightly larger initial step to encourage exploration
        
        steps_without_improvement = 0

        while calls_made[0] < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            # Clip to ensure we stay in bounds, but let the ES handle the rest
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = wrapped_func(y)
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                steps_without_improvement = 0
                # Increase sigma on success, but cap it
                sigma = min(sigma * 1.2, max_sigma)
            else:
                steps_without_improvement += 1
                # Decrease sigma on failure
                sigma = max(sigma * 0.85, min_sigma)

            steps_in_window += 1
            
            # Adapt step size based on 1/5 rule
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > 1/5:
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    sigma = max(sigma * 0.9, min_sigma)
                successes = 0
                steps_in_window = 0

            # Stall detection: if no improvement for a while, restart
            if steps_without_improvement > stall_threshold:
                # Reset local variables for a new run
                # Try a few different restart strategies
                # 1. Random point in [-3, 3]
                # 2. Random point in [-5, 5]
                # 3. Perturb the best known point
                
                # We'll use a simple random restart in a slightly larger box
                # to allow escape from local minima, but biased towards center
                x = rng.uniform(-3, 3, n)
                fx = wrapped_func(x)
                sigma = 1.5
                steps_without_improvement = 0
                successes = 0
                steps_in_window = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
