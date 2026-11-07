"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        call_count = 0

        def timed_func(x):
            nonlocal call_count
            fx = float(func(x))
            call_count += 1
            return fx

        # Strategy: Multi-start with adaptive (1+1)-ES
        # For Rastrigin rotated, the function has many local minima.
        # We use multiple random starts with a budget allocation.

        # Number of restarts: allocate budget across restarts
        # Early restarts get less budget, later ones get more
        # Total budget: 500 calls
        
        # Phase 1: Coarse exploration with many short runs
        # Phase 2: Focused exploitation on best found
        
        # Let's do:
        # - 10 short runs of 30 calls each (300 calls)
        # - 1 long run of 150 calls starting from best found (150 calls)
        # - 50 calls remaining for final polish
        # Total: 500 calls
        
        # Actually, let's be more adaptive:
        # Use a pool of starts, each with a decaying budget
        
        num_restarts = 8
        # Budget allocation: first restart gets more, subsequent get less
        # Total must be <= budget
        
        # Allocate: 60, 55, 50, 45, 40, 35, 30, 25 = 340
        # Remaining 160 for final polish from best
        restart_budgets = [60, 55, 50, 45, 40, 35, 30, 25]
        final_budget = budget - sum(restart_budgets)  # 160
        
        # Ensure we don't exceed budget
        if sum(restart_budgets) + final_budget > budget:
            # Scale down
            scale = budget / (sum(restart_budgets) + final_budget)
            restart_budgets = [int(b * scale) for b in restart_budgets]
            final_budget = budget - sum(restart_budgets)
        
        for r in range(num_restarts):
            if call_count >= budget or time.time() >= deadline:
                break
            
            # Random start
            x = rng.uniform(-5, 5, dim)
            fx = timed_func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            rb = restart_budgets[r]
            sigma = 1.5
            min_sigma = 0.01
            max_sigma = 3.0
            
            for _ in range(rb):
                if call_count >= budget or time.time() >= deadline:
                    break
                
                # Adaptive sigma based on recent success rate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = timed_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.2, max_sigma)
                else:
                    sigma = max(sigma * 0.8, min_sigma)
            
            # If time is running out, break early
            if time.time() >= deadline - 10:
                break
        
        # Final phase: exploit best found with smaller steps
        if best_x is not None and call_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            min_sigma = 0.001
            max_sigma = 2.0
            
            for _ in range(final_budget):
                if call_count >= budget or time.time() >= deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = timed_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    sigma = max(sigma * 0.85, min_sigma)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
