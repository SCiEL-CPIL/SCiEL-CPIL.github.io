"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

class Optimizer:
    """Multi-Start Nelder-Mead optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.all_points = []
        self.all_fvals = []
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise ValueError("Time limit exceeded")
            
            self.call_count += 1
            fx = func(x)
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            
            self.all_points.append(x.copy())
            self.all_fvals.append(fx)
            
            return fx

        # Phase 1: Global Exploration using Halton Sequence (Stratified Random)
        # We use a low-discrepancy sequence to spread points evenly in the box [-5, 5]^d
        # This is more efficient than pure random for global basin identification.
        
        def halton(index, base):
            f = 1
            r = 0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index = index // base
            return r

        num_global = 30
        global_points = []
        for i in range(1, num_global + 1):
            point = np.array([halton(i, b) for b in range(2, 2 + self.dim)])
            # Map from [0, 1] to [-5, 5]
            point = point * 10.0 - 5.0
            global_points.append(point)
        
        # Evaluate global points
        evaluated_global = []
        for p in global_points:
            try:
                fp = wrapped_func(p)
                evaluated_global.append((fp, p))
            except ValueError:
                break

        if not evaluated_global:
            # Fallback if even the first call fails or budget is 0 (unlikely)
            return self.best_x, self.best_f

        # Sort by function value
        evaluated_global.sort(key=lambda x: x[0])
        
        # Select top K distinct starting points for local search
        # We want points that are well-separated to cover different basins
        # Simple heuristic: take the best point, then the next best that is far enough away
        K = 5
        start_points = [evaluated_global[0][1]]
        
        min_dist = 2.0  # Minimum Euclidean distance between start points
        
        for i in range(1, len(evaluated_global)):
            if len(start_points) >= K:
                break
            cand = evaluated_global[i][1]
            is_far = True
            for sp in start_points:
                if np.linalg.norm(cand - sp) < min_dist:
                    is_far = False
                    break
            if is_far:
                start_points.append(cand)
        
        # If we didn't find enough far points, just take the next best ones
        while len(start_points) < K and len(evaluated_global) > len(start_points):
            # Find the next best point not already in start_points
            for fp, p in evaluated_global:
                if p not in start_points: # Note: array comparison in list might be tricky, use a check
                    # Better check:
                    already_in = False
                    for sp in start_points:
                        if np.allclose(p, sp):
                            already_in = True
                            break
                    if not already_in:
                        start_points.append(p)
                        break

        # Phase 2: Local Refinement with Nelder-Mead
        # Allocate remaining budget among the start points.
        # We'll run them sequentially, giving each a portion of the remaining budget.
        # However, Nelder-Mead's internal loop doesn't easily accept a "max func evals" limit 
        # other than maxiter. We rely on the wrapper to raise an exception.
        
        remaining_budget = self.budget - self.call_count
        if remaining_budget <= 0:
            return self.best_x, self.best_f

        # We will try to run Nelder-Mead for each start point.
        # To be fair, we can just run them one by one until budget is exhausted.
        # Since Nelder-Mead converges fast, the first one might use most of the budget.
        # A better strategy: Limit each run to a fraction of the budget?
        # Or just run them sequentially. The best one will likely finish first and find the global min.
        
        # Let's run them sequentially.
        for sp in start_points:
            if self.call_count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Reset the exception state for the next run? 
            # The wrapper raises ValueError. We catch it.
            
            # Initial simplex size
            # Default is 0.05 * x0 or 0.0002 if x0 is 0.
            # For a box of size 10, a step of 1.0 is reasonable.
            initial_simplex_size = 1.0
            
            try:
                res = minimize(
                    wrapped_func, 
                    sp, 
                    method='Nelder-Mead', 
                    options={
                        'initial_simplex_size': initial_simplex_size,
                        'xatol': 1e-8, 
                        'fatol': 1e-8, 
                        'maxiter': 1000,
                        'adaptive': True
                    }
                )
            except ValueError:
                # Budget or time exceeded
                break
            except Exception:
                # Other errors, break to be safe
                break

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
