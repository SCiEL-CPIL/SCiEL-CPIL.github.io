"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multi-start, and polish phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Multi-start exploration (use ~20% of budget)
        n_starts = max(3, budget // 20)
        best_x = None
        best_fx = np.inf
        
        for i in range(n_starts):
            if time.time() > deadline:
                break
            # Spread initial points across the domain
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        calls = n_starts
        x = best_x
        fx = best_fx
        sigma = 1.5  # Start with moderate step size
        sigma_min = 1e-4
        sigma_max = 5.0
        success = 0
        target_success_rate = 0.2  # 1/5 rule
        
        # Phase 2: Main (1+1)-ES with adaptive sigma
        polish_budget = max(10, budget // 25)
        main_budget = budget - calls - polish_budget
        
        for i in range(main_budget):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                success += 1
            else:
                # No improvement
                pass
            
            # Adapt sigma every 5 steps for stability
            if (i + 1) % 5 == 0:
                rate = success / 5.0
                if rate > target_success_rate:
                    sigma *= 1.1
                elif rate < target_success_rate:
                    sigma *= 0.9
                success = 0
                sigma = np.clip(sigma, sigma_min, sigma_max)
        
        # Phase 3: Polish with smaller steps
        sigma_polish = max(sigma * 0.1, 1e-3)
        for i in range(polish_budget):
            if time.time() > deadline:
                break
            # Try multiple small perturbations
            for _ in range(3):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 0.8
                else:
                    sigma_polish *= 0.9
            # Gradually reduce polish step
            sigma_polish *= 0.95
            sigma_polish = max(sigma_polish, 1e-5)
        
        return x, fx
# EVOLVE-BLOCK-END
