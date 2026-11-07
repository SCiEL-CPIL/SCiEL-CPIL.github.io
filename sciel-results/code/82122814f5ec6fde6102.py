"""Optimized optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex method initialized at box center for sharp ridge tracking."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        # Multi-start (1+1)-ES
        # Number of restarts to explore the space
        num_restarts = 50
        
        # Each restart gets a portion of the budget
        # We will dynamically allocate budget based on remaining time and calls
        # But for simplicity and robustness, we use a fixed budget per restart 
        # or stop when global budget is exhausted.
        
        # Strategy: 
        # 1. Pick a random starting point in [-5, 5]^dim
        # 2. Run (1+1)-ES with adaptive step size
        # 3. Stop if budget or time is exhausted
        
        # To ensure we use the full budget effectively, we don't fix the budget per restart,
        # but rather let the loop run until global budget is hit.
        # However, to avoid getting stuck in a bad local minimum for too long,
        # we can limit the number of steps per restart.
        
        # Let's use a dynamic allocation:
        # Total budget 500.
        # If we do 50 restarts, that's 10 calls each on average.
        # But some restarts might be very quick if they find a good basin.
        
        # A better approach for Sharp Ridge:
        # Start from center, then random points.
        
        # Initial point: Center
        x_current = np.zeros(dim)
        f_current = func(x_current)
        calls_made += 1
        best_x = x_current.copy()
        best_fx = f_current
        
        # Adaptive step size
        sigma = 1.0
        sigma_success = 1.5
        sigma_failure = 0.87
        
        # We will perform a sequence of (1+1)-ES steps.
        # To handle "Sharp Ridge", we might need to track the direction of the ridge.
        # But standard (1+1)-ES with adaptive sigma is often robust enough for BBOB f13 
        # if initialized well.
        
        # Let's use a restart strategy:
        # If no improvement for K steps, restart from a random point or best point with larger sigma.
        
        stagnation_limit = 20
        no_improvement_count = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            x_candidate = x_current + sigma * rng.standard_normal(dim)
            x_candidate = np.clip(x_candidate, -5, 5)
            
            if calls_made >= budget:
                break
                
            f_candidate = func(x_candidate)
            calls_made += 1
            
            if f_candidate < f_current:
                # Success
                x_current = x_candidate
                f_current = f_candidate
                sigma *= sigma_success
                no_improvement_count = 0
                
                if f_current < best_fx:
                    best_fx = f_current
                    best_x = x_current.copy()
            else:
                # Failure
                sigma *= sigma_failure
                no_improvement_count += 1
                
                # If stagnating, restart from best known point with a larger step size
                # or a random point
                if no_improvement_count > stagnation_limit:
                    # Restart from best point with larger sigma
                    x_current = best_x.copy()
                    f_current = best_fx
                    sigma = max(sigma * 5.0, 1.0) # Boost sigma to escape local basin
                    no_improvement_count = 0
                    
                    # Optionally, if sigma is too large, clamp it
                    if sigma > 10.0:
                        sigma = 10.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
