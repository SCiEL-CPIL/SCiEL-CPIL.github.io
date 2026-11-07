"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic hierarchical coordinate-wise descent for Katsuura-like f23."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Coarse grid search to identify basin
        # For 5D, we do a coarse grid with ~4 points per dimension initially
        # That's 4^5 = 1024 evaluations, which is too many for 500 budget
        # Instead, we use a smarter approach: coordinate-wise coarse search
        
        # First, find good initial point using coordinate-wise coarse search
        # For each dimension, try several candidate values
        coarse_candidates = np.linspace(-4, 4, 9)  # 9 points per dimension
        
        # Start from center
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Coordinate-wise coarse descent: for each dimension, find best among coarse candidates
        for dim_idx in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            best_val = fx
            best_x = x.copy()
            for c in coarse_candidates:
                if calls >= budget or time.time() > deadline:
                    break
                x[dim_idx] = c
                fx_c = func(x)
                calls += 1
                if fx_c < best_val:
                    best_val = fx_c
                    best_x = x.copy()
            x = best_x
            fx = best_val
        
        # Phase 2: Fine-tuning with local coordinate-wise search
        # Use finer grids around the current best point
        step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125]
        
        for step in step_sizes:
            if calls >= budget or time.time() > deadline:
                break
            improved = True
            while improved and calls < budget and time.time() <= deadline:
                improved = False
                for dim_idx in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    # Try moving in both directions
                    for direction in [-1, 1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        new_x = x.copy()
                        new_x[dim_idx] = np.clip(new_x[dim_idx] + direction * step, -5, 5)
                        if np.allclose(new_x, x):
                            continue
                        fx_new = func(new_x)
                        calls += 1
                        if fx_new < fx:
                            x = new_x
                            fx = fx_new
                            improved = True
                            break  # Move to next dimension
                    # Also try intermediate points for better convergence
                    if calls < budget and time.time() <= deadline:
                        for frac in [0.5, 0.25, 0.75]:
                            if calls >= budget or time.time() > deadline:
                                break
                            new_x = x.copy()
                            old_val = x[dim_idx]
                            new_val = np.clip(old_val + direction * step * frac, -5, 5)
                            if abs(new_val - old_val) < 1e-10:
                                continue
                            new_x[dim_idx] = new_val
                            fx_new = func(new_x)
                            calls += 1
                            if fx_new < fx:
                                x = new_x
                                fx = fx_new
                                improved = True
                                break
                            if fx_new > fx:
                                # Try the other side of the midpoint
                                new_val2 = np.clip(old_val - direction * step * (1 - frac), -5, 5)
                                if abs(new_val2 - old_val) > 1e-10 and abs(new_val2 - new_val) > 1e-10:
                                    new_x2 = x.copy()
                                    new_x2[dim_idx] = new_val2
                                    fx_new2 = func(new_x2)
                                    calls += 1
                                    if fx_new2 < fx:
                                        x = new_x2
                                        fx = fx_new2
                                        improved = True
                                        break
        
        return x, fx
# EVOLVE-BLOCK-END
