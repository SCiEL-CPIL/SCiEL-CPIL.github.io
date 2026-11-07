# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # We'll use a portfolio approach: multiple restarts with (1+1)-ES
        # Each restart gets a portion of the budget
        # Key insight: Katsuura function has many local minima, so diversity is important
        
        best_x = None
        best_f = np.inf
        
        # Number of restarts - we'll use a dynamic approach
        # Start with a few broad explorations, then focus on promising areas
        num_restarts = min(10, max(3, budget // 50))
        calls_per_restart = budget // num_restarts
        calls_per_restart = max(calls_per_restart, 10)
        
        # First, do a quick random search to find some good starting points
        # Then refine each with (1+1)-ES
        
        evaluated_points = []
        
        # Phase 1: Quick random sampling to identify promising regions
        n_sample = min(20, budget // 25)
        for i in range(n_sample):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            evaluated_points.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort by fitness and take top candidates for refinement
        evaluated_points.sort(key=lambda t: t[0])
        
        # Phase 2: Refine top candidates with (1+1)-ES
        top_k = min(5, len(evaluated_points))
        
        for k in range(top_k):
            if time.time() > deadline:
                break
            if len(evaluated_points) <= k:
                break
            
            fx_init, x_init = evaluated_points[k]
            
            # (1+1)-ES refinement
            x = x_init.copy()
            fx = fx_init
            sigma = 1.0  # initial step size
            
            # Adaptive sigma with success rate tracking
            success_count = 0
            window_size = 10
            window = [0] * window_size
            window_idx = 0
            window_full = False
            
            remaining_calls = budget - len(evaluated_points)
            # Cap the number of calls per refinement to leave time for others
            max_refine_calls = min(calls_per_restart, max(10, remaining_calls // (top_k - k)))
            
            for step in range(max_refine_calls):
                if time.time() > deadline:
                    break
                
                # Generate trial point
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                # Update success window
                success = 1 if fy < fx else 0
                window[window_idx] = success
                window_idx = (window_idx + 1) % window_size
                if not window_full and window_idx == 0:
                    window_full = True
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma *= 1.2
                else:
                    # Decrease sigma on failure
                    sigma *= 0.85
                
                # Adapt sigma based on success rate
                if window_full:
                    success_rate = sum(window) / window_size
                    # Target ~1/5 success rate for optimal progress
                    if success_rate > 0.4:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.9
                    
                    # Clamp sigma
                    sigma = np.clip(sigma, 0.01, 5.0)
            
            # After refinement, if we found a better point, add to evaluated
            if fx < best_f or True:
                evaluated_points.append((fx, x.copy()))
        
        # Phase 3: If we still have budget, do more random restarts from the best region
        remaining = budget - len(evaluated_points)
        if remaining > 5 and best_x is not None:
            # Local search around the best point found
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            
            for step in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                sigma = np.clip(sigma, 0.005, 2.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
