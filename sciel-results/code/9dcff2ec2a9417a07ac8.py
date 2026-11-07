"""Multi-start (1+1)-ES for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with global exploration phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Phase 1: Global exploration - evaluate 50 random points
        n_explore = min(50, self.budget // 5)
        candidates = []
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            candidates.append((fx, x))
        
        if not candidates:
            # Fallback: single point
            x = np.zeros(self.dim)
            fx = func(x)
            return x, fx
        
        # Sort by fitness and keep top 5
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))
        
        # Phase 2: Run (1+1)-ES from each of the top starting points
        remaining_budget = self.budget - len(candidates)
        if remaining_budget <= 0:
            best = candidates[0]
            return best[1], best[0]
        
        budget_per_es = max(1, remaining_budget // top_k)
        best_x, best_f = candidates[0][1], candidates[0][0]
        
        for i in range(top_k):
            if time.time() > deadline:
                break
            
            x = candidates[i][1].copy()
            fx = candidates[i][0]
            sigma = 1.0  # smaller initial step for exploitation
            
            # (1+1)-ES with adaptive sigma
            for step in range(budget_per_es):
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                else:
                    sigma *= 0.85
                
                # If sigma becomes too small, try a larger jump
                if sigma < 0.01:
                    sigma = 0.5
        
        return best_x, best_f
# EVOLVE-BLOCK-END
