"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts, step-size adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 30s margin before 310s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0
        remaining = self.budget

        # We'll do multiple restarts. Each restart gets a chunk of the budget.
        # First restart gets a larger chunk to find a good region.
        # Subsequent restarts get smaller chunks for refinement.
        
        # Strategy: 
        # - Use (1+1)-ES with adaptive step size
        # - Restart if no improvement for a while
        # - Use the best point found so far as a starting point for refinement
        # - Keep a small population for diversity in early stages

        # Phase 1: Coarse search with larger steps and restarts
        # Phase 2: Fine search around the best point found

        def run_es(x0, fx0, n_calls, sigma0, sigma_min, sigma_max):
            """Run (1+1)-ES for n_calls evaluations starting from x0."""
            nonlocal calls_made, best_x, best_f
            
            x = x0.copy()
            fx = fx0
            sigma = sigma0
            
            no_improve = 0
            no_improve_limit = max(20, n_calls // 10)
            
            for i in range(n_calls):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Generate trial point
                step = sigma * np.random.randn(self.dim)
                y = np.clip(x + step, -5.0, 5.0)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.2, sigma_max)
                    no_improve = 0
                    
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.9, sigma_min)
                    no_improve += 1
                
                # Restart condition: too many non-improvements
                if no_improve >= no_improve_limit and i < n_calls - 10:
                    break
            
            return x, fx, sigma

        # Initial coarse search: multiple restarts with large steps
        n_coarse = min(200, remaining)
        n_restarts_coarse = 3
        calls_per_restart = n_coarse // n_restarts_coarse

        for r in range(n_restarts_coarse):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            # Start from random point or best found
            if r == 0 or best_x is None:
                x0 = np.random.uniform(-5.0, 5.0, self.dim)
                fx0 = func(x0)
                calls_made += 1
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
            else:
                # Perturb the best point
                x0 = np.clip(best_x + np.random.randn(self.dim) * 1.0, -5.0, 5.0)
                fx0 = func(x0)
                calls_made += 1
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()
            
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            x, fx, sigma = run_es(
                x0, fx0, 
                calls_per_restart - 1,  # -1 because we already evaluated x0
                2.0, 0.1, 5.0
            )
        
        # Phase 2: Fine search around the best point
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining2 = self.budget - calls_made
            # Use smaller steps for refinement
            x0 = best_x.copy()
            fx0 = best_f
            run_es(x0, fx0, remaining2, 0.5, 0.01, 2.0)
        
        # Phase 3: If we have time and budget left, do another round of refinement
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining3 = self.budget - calls_made
            if remaining3 > 50:
                x0 = best_x.copy()
                fx0 = best_f
                run_es(x0, fx0, remaining3, 0.1, 0.001, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
