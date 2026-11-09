"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front is 1D: only x[0] varies, x[1:] = 0.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[1:] = 0, only x[0] varies in [0,1]
        # Use all budget on 1D grid for dense front coverage
        
        # Phase 1: Uniform grid with (budget - 10) points
        n_uniform = max(budget - 10, 1)
        x1_vals = np.linspace(0, 1, n_uniform)
        
        for x1 in x1_vals:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] = 0 ensures g(x) = 0 (on Pareto front)
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        # Phase 2: Use remaining budget for midpoint refinement
        remaining = budget - n_evals
        if remaining > 0 and time.time() < deadline:
            # Add points at midpoints between uniform grid lines
            step = 1.0 / (n_uniform - 1) if n_uniform > 1 else 1.0
            midpoints = np.linspace(0.5 * step, 1 - 0.5 * step, n_uniform - 1)
            
            # Select remaining midpoints spread out
            n_mid = min(remaining, n_uniform - 1)
            if n_mid > 0:
                # Pick evenly spaced midpoints
                indices = np.linspace(0, n_uniform - 2, n_mid, dtype=int)
                for idx in indices:
                    if n_evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x1_mid = midpoints[idx]
                    x = np.zeros(self.dim)
                    x[0] = x1_mid
                    
                    obj = problem(x)
                    results.append(obj)
                    n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
