"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer for BBOB f7 (step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with a small population
        # Use a simple population-based search to find good basins
        pop_size = 10
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        
        # Evaluate a diverse set of initial points
        candidates = [x_best.copy()]
        for i in range(min(pop_size - 1, budget // 10)):
            # Mix of random and structured points
            if i % 3 == 0:
                x_cand = rng.uniform(-5, 5, dim)
            elif i % 3 == 1:
                x_cand = rng.uniform(-3, 3, dim)
            else:
                x_cand = rng.uniform(-1, 1, dim)
            
            f_cand = func(x_cand)
            candidates.append((x_cand, f_cand))
            if f_cand < f_best:
                f_best = f_cand
                x_best = x_cand.copy()
            
            if time.time() > deadline:
                break

        # Phase 2: Local refinement with (1+1)-ES
        # Adaptive step size that handles step functions
        x = x_best.copy()
        fx = f_best
        sigma = 1.0
        min_sigma = 1e-4
        max_sigma = 5.0
        
        # Track success for adaptive sigma
        successes = 0
        window = 10
        sigma_history = []
        
        calls_made = len(candidates) + 1  # initial + candidates
        
        for i in range(budget - calls_made):
            if time.time() > deadline:
                break
            
            # Generate perturbation
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx - 1e-10:
                x, fx = y, fy
                successes += 1
                sigma_history.append(1)
            else:
                sigma_history.append(0)
            
            # Update sigma based on success rate in window
            if len(sigma_history) >= window:
                success_rate = sum(sigma_history[-window:]) / window
                if success_rate > 0.5:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.7
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, min_sigma, max_sigma)
            
            # Periodic diversification to escape local minima
            if i > 0 and i % 50 == 0 and sigma < 0.5:
                # Try a larger jump
                y_jump = np.clip(x + 2.0 * rng.standard_normal(dim), -5, 5)
                fy_jump = func(y_jump)
                calls_made += 1
                if fy_jump < fx - 1e-10:
                    x, fx = y_jump, fy_jump
                    sigma = max(sigma, 1.0)
        
        return x, fx
# EVOLVE-BLOCK-END
