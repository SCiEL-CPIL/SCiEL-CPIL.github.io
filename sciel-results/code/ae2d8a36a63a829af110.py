"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-start initialization and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        rng = np.random.default_rng(0)
        
        # Phase 1: Multi-start initialization
        # Sample a few points to find a good starting basin
        n_init = 10
        init_budget = min(n_init, budget)
        
        x_best = None
        f_best = float('inf')
        calls = 0
        
        for _ in range(init_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        if x_best is None:
            # Fallback if no calls were made (should not happen)
            x_best = rng.uniform(-5, 5, dim)
            f_best = func(x_best)
            calls += 1

        # Phase 2: (1+1)-ES refinement
        x_curr = x_best.copy()
        fx_curr = f_best
        sigma = 1.0
        stall_count = 0
        stall_limit = 20

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x_curr + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < f_best:
                # New global best
                f_best = fy
                x_best = y.copy()
                x_curr = y.copy()
                fx_curr = fy
                # Increase sigma slightly to explore further
                sigma *= 1.1
                stall_count = 0
            else:
                # No improvement
                # Decrease sigma to refine
                sigma *= 0.95
                stall_count += 1
                
                # Anti-stagnation: if stuck for a while, kick sigma
                if stall_count > stall_limit:
                    sigma *= 2.0
                    stall_count = 0
                    # Optional: restart from global best if current is far
                    # For now, just kick the sigma

            # Keep sigma within bounds
            sigma = np.clip(sigma, 1e-4, 10.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
