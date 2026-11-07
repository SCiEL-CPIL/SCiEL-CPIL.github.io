"""Multi-start (1+1)-ES with per-dimension step sizes for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with per-dimension step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin
        rng = np.random.default_rng(0)
        
        # Phase 1: Global exploration
        # Use more points to ensure we find the basin
        n_explore = min(100, self.budget // 3)
        candidates = []
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            candidates.append((fx, x))
        
        if not candidates:
            x = np.zeros(self.dim)
            fx = func(x)
            return x, fx
        
        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_f = candidates[0][0]
        
        # Phase 2: Local optimization with (1+1)-ES
        remaining_budget = self.budget - len(candidates)
        if remaining_budget <= 0:
            return best_x, best_f

        x = best_x.copy()
        fx = best_f
        # Initialize step sizes: moderate size
        s = np.ones(self.dim) * 0.5
        no_improve_count = 0
        max_no_improve = 30
        
        for step in range(remaining_budget):
            if time.time() > deadline:
                break
            
            # Generate candidate
            noise = rng.standard_normal(self.dim)
            y = x + s * noise
            y = np.clip(y, -5, 5)
            fy = func(y)
            
            if fy < fx:
                x = y
                fx = fy
                no_improve_count = 0
                # Increase step size
                s *= 1.1
                # Cap step size to avoid leaving the basin
                s = np.clip(s, 0.001, 5.0)
                
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
            else:
                no_improve_count += 1
                # Decrease step size
                s *= 0.9
                s = np.clip(s, 0.001, 5.0)
                
                # Restart logic: if stuck, try a larger jump or reset
                if no_improve_count > max_no_improve:
                    # Reset step sizes to a larger value to escape local flatness
                    s = np.ones(self.dim) * 1.0
                    no_improve_count = 0
                    # Optionally, perturb x slightly? 
                    # For sharp ridge, jumping might be bad. 
                    # Instead, just increase step size and hope it finds the ridge.
                    # Or, restart from a random point near current best?
                    # Let's try a small perturbation from the best known point
                    if rng.random() < 0.1:
                        x = best_x + 0.1 * rng.standard_normal(self.dim)
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        if fx < best_f:
                            best_x = x.copy()
                            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
