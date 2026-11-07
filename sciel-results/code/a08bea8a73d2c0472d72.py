"""Optimizer for BBOB-style f15 (Rastrigin rotated) with Multi-start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        # Multi-start strategy with 6 diverse restarts
        # Including origin and varied uniform ranges to cover more basins
        n_restarts = 6
        
        # Define diverse starting points
        start_points = []
        
        # 1. Origin
        start_points.append(np.zeros(dim))
        
        # 2-6. Random points in different ranges
        # Wider range first, then narrower
        ranges = [(-5, 5), (-4, 4), (-3, 3), (-2, 2), (-1, 1)]
        for lo, hi in ranges:
            start_points.append(rng.uniform(lo, hi, dim))
        
        # Shuffle to mix up the order, but keep origin as one of them
        # Actually, let's just use them in order but ensure diversity
        # We'll allocate budget roughly equally, but maybe give more to later refinements
        
        # Strategy: 
        # Phase 1: Quick exploration of each start point with a small budget
        # Phase 2: Refine the best few points found
        # Phase 3: Final polish on the global best
        
        # Let's do 6 restarts. 
        # Allocate 50% of budget for exploration (1/6 each ~ 8.3%)
        # Allocate 30% for refining the top 2
        # Allocate 20% for final polish
        
        # Simpler approach: Run 6 (1+1)-ES loops with equal budget, but stop early if stuck?
        # The current program uses equal budget. Let's stick to that but ensure the start points are diverse.
        
        calls_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            # Check time and budget
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Initialize restart point
            x = start_points[r].copy()
            
            # Evaluate initial point
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES loop for this restart
            sigma = 2.0  # Initial step size
            # Adjust initial sigma based on the range? 
            # For origin, maybe start smaller? No, 2.0 is fine for [-5,5].
            
            max_calls_this_restart = calls_per_restart - 1
            if max_calls_this_restart < 0:
                continue

            for _ in range(max_calls_this_restart):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    # Success
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure
                    sigma *= 0.87

        # If we still have budget, we can do a final refinement on the best point found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5 # Smaller sigma for refinement
            remaining = budget - calls_made
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2 # Less aggressive growth for refinement
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
