"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size, time guard, and restart logic."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        # Phase 1: Initial random sampling to find a good starting point
        # Use ~10% of budget for initial exploration
        init_budget = min(50, budget // 10)
        for _ in range(init_budget):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if no evaluations were made
            x = np.zeros(dim)
            fx = func(x)
            calls = 1
            best_f = fx
            best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma and restarts
        # We'll run multiple restarts if we stall
        max_restarts = 5
        stall_threshold = 30  # No improvement in 30 steps -> restart
        
        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            if restart == 0:
                x = best_x.copy()
                sigma = 1.0
            else:
                # Perturb best point for restart
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                sigma = 1.0
            
            # Evaluate current x if not already best
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            stall_counter = 0
            prev_best = best_f
            
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2  # Slightly more conservative growth
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        stall_counter = 0
                    else:
                        stall_counter += 1
                else:
                    sigma *= 0.9  # Shrink step size
                    stall_counter += 1
                
                if stall_counter >= stall_threshold:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
