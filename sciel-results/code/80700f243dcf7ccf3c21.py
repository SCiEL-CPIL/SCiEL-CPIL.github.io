"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with time guard for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        dim = self.dim
        budget = self.budget

        def _es_run(rng, x0, f0, remaining_budget):
            nonlocal calls_used, best_x, best_f
            x = x0.copy()
            fx = f0
            sigma = 2.0
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    return x, fx
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85
            return x, fx

        # Phase 1: Several random restarts with moderate budgets
        n_restarts = 5
        base_budget = budget // n_restarts
        for i in range(n_restarts):
            if time.time() > deadline:
                break
            rng = np.random.default_rng(1000 + i * 100)
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f, best_x = f0, x0.copy()
            
            remaining = min(base_budget, budget - calls_used)
            if remaining <= 0:
                break
            _es_run(rng, x0, f0, remaining)

        # Phase 2: Continue from best found so far with remaining budget
        if best_x is not None and calls_used < budget:
            if time.time() < deadline:
                rng = np.random.default_rng(9999)
                remaining = budget - calls_used
                _es_run(rng, best_x, best_f, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
