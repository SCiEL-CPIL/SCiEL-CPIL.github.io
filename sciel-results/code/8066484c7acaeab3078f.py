"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with time guard and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        rng = np.random.default_rng(0)
        
        # Track best solution found
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return None
            if time.time() >= deadline:
                return None
            calls_made += 1
            return func(x)
        
        # Multi-start: try several initial points
        n_starts = 5
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Random points in [-4, 4]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))
        
        # Evaluate initial points
        for x0 in starts:
            if time.time() >= deadline or calls_made >= self.budget:
                break
            f0 = safe_call(x0)
            if f0 is not None and f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # For each start, run a (1+1)-ES
        for x0 in starts:
            if time.time() >= deadline or calls_made >= self.budget:
                break
            
            # Skip if we already have a good solution and this start is worse
            # Actually, let's just run ES from each start point
            x = x0.copy()
            fx = safe_call(x)
            if fx is None:
                continue
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0
            # Run (1+1)-ES with adaptive sigma
            while calls_made < self.budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                
                # Stop if sigma is too small
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
