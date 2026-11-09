import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # Phase 1: Global Exploration
        # Use 20% of budget for random sampling to find good basins
        n_explore = max(20, int(self.budget * 0.2))
        points = np.random.uniform(-5, 5, size=(n_explore, self.dim))
        values = np.array([func(p) for p in points])
        
        # Keep track of best
        best_val = np.min(values)
        best_idx = np.argmin(values)
        best_x = points[best_idx]
        
        # Sort indices by value to get top candidates
        sorted_indices = np.argsort(values)
        
        # Phase 2: Local Exploitation with Nelder-Mead
        # We will try to refine the top candidates
        # Reserve some budget for exploration, use rest for refinement
        remaining_budget = self.budget - n_explore
        
        # We'll try up to 3 best candidates from exploration
        # Allocate remaining budget roughly equally, but prioritize the best
        candidates = []
        for idx in sorted_indices[:3]:
            # Ensure candidates are distinct enough
            is_distinct = True
            for cx in candidates:
                if np.linalg.norm(points[idx] - cx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                candidates.append(points[idx])
        
        # If no candidates found (shouldn't happen), use best
        if not candidates:
            candidates = [best_x]
        
        # Allocate budget to candidates
        # Give more budget to the best candidate
        n_candidates = len(candidates)
        # Simple allocation: 50% to best, 25% to 2nd, 25% to 3rd (if exist)
        # Or just split evenly for simplicity, but best first is better for AOCC
        # Let's give 60% to the first, 20% to second, 20% to third
        weights = []
        if n_candidates == 1:
            weights = [1.0]
        elif n_candidates == 2:
            weights = [0.6, 0.4]
        else:
            weights = [0.5, 0.25, 0.25]
            
        allocs = [int(remaining_budget * w) for w in weights]
        
        # Run Nelder-Mead on each candidate
        for i, start_point in enumerate(candidates):
            if time.time() > deadline:
                break
                
            if allocs[i] <= 0:
                continue
                
            # Nelder-Mead options
            # maxiter is the max number of iterations, not function calls
            # In 5D, 1 iteration ~ 5-10 function calls
            # So maxiter ~ allocs[i] / 5
            max_iter = max(10, int(allocs[i] / 5))
            
            try:
                res = minimize(
                    func, 
                    start_point, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass
            
            # Update remaining budget? 
            # The minimize function doesn't report exact calls used easily in all versions, 
            # but we can assume it used up to max_iter * dim approx.
            # For safety, we just proceed to next candidate if time allows.
            
        return best_x, best_val
