"""Optimizer for BBOB-style f15 (Rastrigin rotated) with Multi-start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Use a fixed seed for reproducibility within the run, or random?
        # The task says "hidden seeded shift", but our optimizer should be deterministic or at least consistent.
        # Using a fixed seed for the RNG inside the optimizer ensures consistent behavior if the evaluator runs multiple times,
        # but typically evaluators might want different trajectories? No, the evaluator calls __call__ once per instance.
        # The instances have hidden seeds. Our optimizer should just be a good black box.
        # Let's use a fixed seed to ensure the "search strategy" is stable, or np.random for diversity?
        # Inspiration 1 used rng = np.random.default_rng(0). Let's stick to that for consistency with the high-scoring model.
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Multi-start strategy
        # We divide the budget into several restarts.
        # Rastrigin has many local minima. A single (1+1)-ES might get stuck.
        # Let's try 5-10 restarts.
        # The (1+1)-ES with 1/5 rule is fast.
        # Let's allocate budget such that each restart has enough to converge locally.
        # If we do 5 restarts, each gets budget/5.
        
        n_restarts = 5
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            # Check time and budget
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Initialize restart point
            # Start from center for the first restart? Or random?
            # Random is better for multimodal functions.
            # Let's use uniform random in [-5, 5].
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES loop for this restart
            sigma = 2.0  # Initial step size
            # We can adjust sigma based on the success rate, but 1/5 rule is standard.
            # Success: sigma *= 1.5
            # Failure: sigma *= 0.87
            
            max_calls_this_restart = calls_per_restart - 1 # 1 call for init
            if max_calls_this_restart < 0:
                continue

            for _ in range(max_calls_this_restart):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure
                    sigma *= 0.87

        # If we still have budget, we can do a final refinement on the best point found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Smaller sigma for refinement
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2 # Less aggressive growth for refinement
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
