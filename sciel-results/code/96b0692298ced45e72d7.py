import numpy as np
import time

class Optimizer:
    """(1+1)-ES with adaptive sigma and manual local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        # Phase 1: (1+1)-ES for most of the budget
        # Reserve last 30 calls for local refinement
        polish_budget = 30
        es_budget = budget - polish_budget

        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.4
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.85

            # Clamp sigma
            if sigma > 8.0:
                sigma = 8.0
            elif sigma < 1e-12:
                sigma = 1e-12

        # Phase 2: Manual local refinement using remaining budget
        remaining = budget - calls
        if remaining > 2 and time.time() < deadline:
            # Start from best point, use very small steps
            x_local = best_x.copy()
            fx_local = best_fx
            sigma_local = max(sigma * 0.1, 1e-4)
            
            for _ in range(remaining - 1):
                if time.time() >= deadline:
                    break
                if calls >= budget:
                    break
                
                # Try a small random step
                y_local = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
                fy_local = func(y_local)
                calls += 1
                
                if fy_local < fx_local:
                    x_local, fx_local = y_local, fy_local
                    sigma_local *= 1.2
                    if fx_local < best_fx:
                        best_x, best_fx = x_local.copy(), fx_local
                else:
                    sigma_local *= 0.7
                
                if sigma_local < 1e-12:
                    break

        return best_x, best_fx
