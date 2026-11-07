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
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Strategy: Hybrid approach.
        # 1. Quick exploration phase with several diverse restarts to find good basins.
        # 2. Deep refinement phase on the best found point using aggressive (1+1)-ES.
        # This combines the basin-finding capability of multi-start (inspiration 2)
        # with the deep convergence of the single-lineage approach (current program).
        
        # Phase 1: Exploration
        # Use a fraction of the budget to explore several starting points.
        # 30% of budget for exploration, 70% for refinement.
        # If exploration finds a very good point, we can spend more time refining it.
        
        exploration_budget = int(budget * 0.3)
        refinement_budget = budget - exploration_budget
        
        # Define diverse starting points for exploration
        # Origin is often a good attractor or close to one.
        # Random points in various ranges cover the space.
        start_points = []
        start_points.append(np.zeros(dim))
        
        # Add random points in different ranges
        ranges = [(-5, 5), (-4, 4), (-3, 3), (-2, 2), (-1, 1)]
        for lo, hi in ranges:
            start_points.append(rng.uniform(lo, hi, dim))
        
        # Shuffle start points to ensure diversity in order, but keep origin as first?
        # Actually, let's just iterate through them.
        # Allocate equal budget for each exploration restart.
        n_restarts = len(start_points)
        calls_per_restart = exploration_budget // n_restarts
        if calls_per_restart < 10:
            calls_per_restart = 10
            
        # Ensure we don't exceed exploration budget
        total_exploration_calls = 0
        
        for r in range(n_restarts):
            if calls_made >= exploration_budget or time.time() > deadline:
                break
            
            x = start_points[r].copy()
            fx = func(x)
            calls_made += 1
            total_exploration_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # (1+1)-ES loop for this restart
            sigma = 2.0
            max_calls_this_restart = calls_per_restart - 1
            if max_calls_this_restart < 0:
                continue
                
            for _ in range(max_calls_this_restart):
                if calls_made >= exploration_budget or time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                total_exploration_calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87

        # Phase 2: Refinement
        # If we have a best_x, use the remaining budget to refine it deeply.
        # Use a more aggressive sigma adaptation to escape shallow local minima
        # and converge faster to the global minimum if we are in the correct basin.
        
        if best_x is not None and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0  # Start with moderate sigma
            
            # If the best_f is already very low, we can be more conservative.
            # But for Rastrigin, we might need to jump out of local minima.
            # Let's use the same aggressive adaptation as the current program.
            
            remaining_budget = budget - calls_made
            
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87
                    
                # Early termination if sigma becomes very small and we are stuck
                if sigma < 1e-8:
                    break

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
