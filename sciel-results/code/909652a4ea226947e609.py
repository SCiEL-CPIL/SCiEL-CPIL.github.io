"""Bootstrap portfolio: multi-restart population-based search with adaptive step sizes."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio search:
    1. Brief exploratory phase with many random restarts to find promising basins.
    2. Intensive local refinement using CMA-ES-like logic (diagonal covariance)
       or simple adaptive step-size (1+1)-ES with success/failure scaling.
    3. Time-budgeted: keeps the global best and uses remaining time for further restarts/refinements.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit

        # Global best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return time.time() >= deadline or calls_made >= budget

        # Phase 1: Exploratory random sampling to find promising basins
        # Use a small population of random points
        rng = np.random.default_rng(42)
        
        # Do a quick random search to find a good starting point
        n_explore = min(50, budget // 10)
        explore_x = rng.uniform(-5, 5, (n_explore, dim))
        explore_f = np.array([evaluate(x) for x in explore_x])
        
        # Sort by fitness and keep the best few as seeds
        sort_idx = np.argsort(explore_f)
        seeds = []
        for idx in sort_idx:
            if len(seeds) >= 5:
                break
            if explore_f[idx] < best_f * 0.9 or len(seeds) < 3:  # Keep diverse good points
                seeds.append(explore_x[idx].copy())

        # If no seeds found (shouldn't happen), use the best point
        if not seeds:
            seeds = [explore_x[sort_idx[0]].copy()]

        # Phase 2: Local refinement with adaptive step size
        # For each seed, run a (1+1)-ES with adaptive sigma
        for seed in seeds:
            if check_deadline():
                break
                
            x = seed.copy()
            fx = best_f if np.allclose(x, best_x) else evaluate(x)
            sigma = 0.5  # Start with smaller step size for local refinement
            
            # Run (1+1)-ES for a limited number of steps
            steps = 0
            max_steps = 50  # Limit steps per seed to allow multiple seeds
            
            while steps < max_steps and not check_deadline():
                # Generate perturbation
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                steps += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3  # Increase step size on success
                else:
                    sigma *= 0.8  # Decrease step size on failure
                
                # Prevent sigma from getting too small or too large
                sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 3: If we still have budget, do a final intensive search around the best point
        if not check_deadline() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Very fine-grained search
            
            # Run for remaining budget
            while not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
