"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Phase 1: Global search with periodic-aware sampling
        # Rastrigin has minima at integer coordinates (after shift/rotation)
        # We sample around integer grid points and also random points
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        max_calls = self.budget

        # Strategy: Mix of:
        # 1. Integer grid exploration (coarse)
        # 2. Random sampling
        # 3. Local refinement around best found
        
        # First, do a coarse grid search around integers
        # The function is Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Minima occur where x_i is near an integer (in the shifted/rotated space)
        
        # Since we don't know the shift/rotation, we'll use a combination of:
        # - Random points
        # - Points snapped to nearby integers
        # - Local refinement
        
        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse global search (first ~60% of budget)
        phase1_budget = int(0.6 * max_calls)
        
        for i in range(phase1_budget):
            if time.time() > deadline - 60:
                break
            
            # Mix of strategies
            r = rng.random()
            
            if r < 0.4:
                # Random point
                x = rng.uniform(-5, 5, self.dim)
            elif r < 0.7:
                # Snap to nearby integers with small perturbation
                x_base = np.round(rng.uniform(-5, 5, self.dim))
                x = x_base + rng.normal(0, 0.3, self.dim)
            elif r < 0.9:
                # Integer grid with one coordinate perturbed
                x_base = np.round(rng.uniform(-5, 5, self.dim))
                idx = rng.integers(0, self.dim)
                x = x_base.copy()
                x[idx] = x_base[idx] + rng.uniform(-0.5, 0.5)
            else:
                # Perturb current best
                if best_x is not None:
                    x = best_x + rng.normal(0, 1.0, self.dim)
                else:
                    x = rng.uniform(-5, 5, self.dim)
            
            evaluate(x)

        # Phase 2: Local refinement around best point (remaining budget)
        if best_x is not None and calls_used < max_calls:
            # Use a (1+1)-ES style local search
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            remaining = max_calls - calls_used
            for i in range(remaining):
                if time.time() > deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if x is not best_x or fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.3, 5.0)
                else:
                    sigma = max(sigma * 0.85, 0.01)
            
            # Final check
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 3: If we have budget left, try a few more restarts from integer points
        if calls_used < max_calls and best_x is not None:
            remaining = max_calls - calls_used
            # Try refining from a few integer-nearby points
            for i in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                
                # Pick a random integer point near the best
                x0 = np.round(best_x + rng.normal(0, 1.5, self.dim))
                x0 = np.clip(x0, -5, 5)
                
                # Quick local search
                sigma = 0.3
                for j in range(10):
                    if calls_used >= max_calls or time.time() > deadline:
                        break
                    y = np.clip(x0 + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_f:
                        x0 = y.copy()
                        best_f = fy
                        best_x = y.copy()
                        sigma = min(sigma * 1.2, 1.0)
                    else:
                        sigma = max(sigma * 0.9, 0.05)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
