"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import scipy.stats.qmc as qmc

class BudgetExceeded(Exception):
    pass

class Optimizer:
    """Multi-start Nelder-Mead optimizer with Sobol global search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        # Phase 1: Global Search using Sobol Sequence
        n_global = 50
        rng = np.random.default_rng(42)
        
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            points = sampler.random(n_global)
            points = points * 10.0 - 5.0
        except Exception:
            points = np.random.uniform(-5, 5, (n_global, dim))
            
        best_x_global = None
        best_f_global = float('inf')
        candidates = []
        
        for x in points:
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x_clipped.copy()
            candidates.append((fx, x_clipped.copy()))
            
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Multi-start Local Refinement
        top_k = 5
        top_candidates = [x for _, x in candidates[:top_k]]
        
        best_x_final = best_x_global
        best_f_final = best_f_global
        
        total_calls_so_far = n_global
        
        for i in range(top_k):
            start_x = top_candidates[i]
            
            def wrapped_func(x):
                nonlocal total_calls_so_far
                if total_calls_so_far >= total_budget:
                    raise BudgetExceeded()
                total_calls_so_far += 1
                return func(x)
                
            try:
                res = minimize(
                    wrapped_func, start_x, method='Nelder-Mead',
                    options={
                        'maxiter': 1000, 
                        'xatol': 1e-8, 
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f_final:
                    best_f_final = res.fun
                    best_x_final = res.x.copy()
            except BudgetExceeded:
                break
            except Exception:
                pass
                
        return best_x_final, best_f_final
# EVOLVE-BLOCK-END
