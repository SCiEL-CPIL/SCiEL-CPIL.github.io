"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES for BBOB f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start initialization - sample 50 points
        n_init = 50
        init_points = []
        init_vals = []
        
        for _ in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            init_points.append(x)
            init_vals.append(fx)
        
        # Find top 5 starting points
        sorted_indices = np.argsort(init_vals)
        top_k = min(5, len(sorted_indices))
        top_starts = [init_points[i] for i in sorted_indices[:top_k]]
        top_vals = [init_vals[i] for i in sorted_indices[:top_k]]
        
        best_x = top_starts[0].copy()
        best_f = top_vals[0]
        
        # Phase 2: Run (1+1)-ES from each top start point
        remaining_budget = self.budget - n_init
        per_start_budget = max(1, remaining_budget // top_k)
        
        for start_idx in range(top_k):
            if time.time() >= deadline:
                break
            
            x = top_starts[start_idx].copy()
            fx = top_vals[start_idx]
            sigma = 1.0
            
            for step in range(per_start_budget):
                if time.time() >= deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
