"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like optimizer for Rastrigin-rotated functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Phase 1: Multi-start with coarse grid + random restarts
        # For Rastrigin, we want to explore the landscape broadly first
        # Use a set of diverse starting points
        
        n_starts = 15
        start_points = []
        
        # Add center and corners
        start_points.append(np.zeros(self.dim))
        for i in range(2 ** self.dim):
            point = np.array([(i >> j) & 1 for j in range(self.dim)], dtype=float)
            point = point * 8 - 4  # map to [-4, 4]
            start_points.append(point)
        
        # Add random points
        while len(start_points) < n_starts:
            start_points.append(rng.uniform(-4, 4, self.dim))
        
        # Shuffle start points
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx]
        
        for sp in start_points:
            if time.time() > deadline:
                break
            if calls >= self.budget:
                break
                
            x = np.clip(sp, -5, 5)
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Local refinement from this start point using (1+1)-ES
            sigma = 1.0
            local_calls = 0
            max_local_calls = max(10, (self.budget - calls) // (n_starts - len(start_points) + 1))
            
            while local_calls < max_local_calls and calls < self.budget:
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                local_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.75
                    
                # If sigma gets too small, break and try next start
                if sigma < 0.01:
                    break
        
        # Phase 2: Final polish on the best solution found
        if best_x is not None and calls < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = self.budget - calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
        
        return best_x if best_x is not None else np.zeros(self.dim), best_f
# EVOLVE-BLOCK-END
