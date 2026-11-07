"""Deterministic hierarchical grid search for Katsuura-like f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic hierarchical coordinate-wise grid search.
    
    Exploits the separable structure of the Katsuura function.
    Phase 1: Coarse grid search per dimension to locate the basin.
    Phase 2: Fine grid search per dimension to refine the minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0

        def evaluate(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        # Initialize at center
        x = np.zeros(dim)
        best_f = float('inf')
        best_x = x.copy()

        # Phase 1: Coarse Grid Search
        # Use 50 points per dimension across [-5, 5]
        n_coarse = 50
        step_coarse = 10.0 / (n_coarse - 1)
        
        for d in range(dim):
            if calls >= budget:
                break
            local_best_val = x[d]
            local_best_f = float('inf')
            
            for i in range(n_coarse):
                if calls >= budget:
                    break
                v = -5.0 + i * step_coarse
                x[d] = v
                f_val = evaluate(x)
                
                if f_val < local_best_f:
                    local_best_f = f_val
                    local_best_val = v
                    
            x[d] = local_best_val
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = x.copy()

        # Phase 2: Fine Grid Search
        # Refine around the best values from Phase 1
        # Use 50 points in a window of size 2 * step_coarse
        n_fine = 50
        window = step_coarse
        
        for d in range(dim):
            if calls >= budget:
                break
            center = x[d]
            lo = center - window
            hi = center + window
            
            # Clip to valid bounds
            if lo < -5.0:
                lo = -5.0
            if hi > 5.0:
                hi = 5.0
                
            if lo >= hi:
                continue
                
            step_fine = (hi - lo) / (n_fine - 1)
            local_best_val = center
            local_best_f = float('inf')
            
            for i in range(n_fine):
                if calls >= budget:
                    break
                v = lo + i * step_fine
                x[d] = v
                f_val = evaluate(x)
                
                if f_val < local_best_f:
                    local_best_f = f_val
                    local_best_val = v
                    
            x[d] = local_best_val
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
