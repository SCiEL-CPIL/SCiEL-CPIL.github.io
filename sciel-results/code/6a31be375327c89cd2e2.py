"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Wrapper to track calls and enforce deadline
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                return best_f

        # 1. Initial Grid / Random Start
        # f22 is multi-modal. We need to find the global basin.
        # A sparse random grid is effective for 5D.
        # Let's use 20 random points spread across the domain.
        n_starts = 20
        rng = np.random.default_rng(42)
        
        starts = []
        # Add center
        starts.append(np.zeros(dim))
        
        # Add random points
        for _ in range(n_starts - 1):
            x = rng.uniform(-5, 5, size=dim)
            starts.append(x)
            
        # Evaluate starts
        for x in starts:
            if calls >= budget or time.time() > deadline:
                break
            f_val = wrapped_func(x)
        
        # Sort starts by fitness
        # We need to store the values. wrapped_func doesn't return a list easily for sorting 
        # without re-evaluating or keeping a dict. 
        # Let's just keep the best one found in the grid phase as the primary candidate,
        # and maybe a few others if budget allows.
        # Actually, let's collect (f, x) pairs manually for the grid phase to sort them.
        
        # Reset for a cleaner implementation of the multi-start logic
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        candidates = []
        
        # Phase 1: Random Search to find basins
        # Use a larger number of random points to ensure we hit the global basin.
        # 500 budget total. Let's spend 100 on random search.
        n_random = 100
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Refinement with L-BFGS-B
        # Pick the top K candidates. 
        # Each L-BFGS-B run will use remaining budget.
        # We don't know exactly how many calls L-BFGS-B will take, but it's usually efficient.
        # Let's try to refine the top 5 candidates.
        top_k = 5
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = candidates[i][1]
            
            # Check if we have enough budget for a reasonable run
            if budget - calls < 10:
                break
                
            try:
                res = minimize(
                    eval_and_track, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'maxls': 20} # Limit iterations to prevent infinite loops
                )
            except Exception:
                pass
                
            # Update best if L-BFGS-B found something better
            # eval_and_track already updates best_f and best_x
            
        return best_x, best_f
# EVOLVE-BLOCK-END
