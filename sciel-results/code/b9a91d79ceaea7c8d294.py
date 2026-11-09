"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Strategy:
        # 1. Generate a diverse set of starting points.
        # 2. Run a local CMA-ES (or simplified adaptive ES) from each.
        # 3. Keep track of the global best.
        
        # Given Weierstrass is highly multimodal, a single start is risky.
        # Grid search is deterministic but might miss the basin if the shift is unlucky.
        # Random multi-start is more robust.
        
        # Let's use a population-based approach.
        # Since budget is 500, we can afford ~10-20 starts with ~25-50 evals each.
        # Or a single CMA-ES with a larger population? 
        # Standard CMA-ES needs population size N ~ 4 + 3*log(dim). For dim=5, N ~ 10.
        # 500 / 10 = 50 generations. This is quite limited for CMA-ES to converge from scratch.
        
        # Alternative: Multi-start (1+1)-ES with better initialization.
        # The previous program used a grid. Let's try random multi-start.
        
        num_starts = 20
        budget_per_start = max(10, (budget - num_starts) // num_starts)
        
        global_best_x = None
        global_best_f = np.inf

        # Pre-generate start points to ensure diversity
        # Mix of uniform random in [-4, 4] (where optimum is) and some in [-5, 5]
        starts = []
        for _ in range(num_starts):
            # 50% in [-4, 4], 50% in [-5, 5]
            if rng.random() < 0.5:
                x0 = rng.uniform(-4.0, 4.0, dim)
            else:
                x0 = rng.uniform(-5.0, 5.0, dim)
            starts.append(x0)

        for i in range(num_starts):
            if time.time() > deadline:
                break
            if budget <= 0:
                break

            x = starts[i].copy()
            fx = func(x)
            budget -= 1
            
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()

            # Local refinement from this start
            sigma = 1.0
            local_budget = budget_per_start
            no_improve_count = 0
            max_no_improve = 20 # Restart logic within local? No, just stop or reduce sigma.
            
            while local_budget > 0 and budget > 0:
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = func(y)
                budget -= 1
                local_budget -= 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    no_improve_count = 0
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve_count += 1
                    if no_improve_count > 30:
                        # Stagnated locally, break to try next start
                        break

        if global_best_x is None:
            # Fallback
            x = np.zeros(dim)
            fx = func(x)
            return x, fx

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
