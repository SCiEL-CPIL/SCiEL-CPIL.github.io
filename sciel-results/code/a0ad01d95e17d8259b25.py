"""Optimized black-box continuous optimizer for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol


class Optimizer:
    """Systematic Sobol sampling + Multi-Start Nelder-Mead refinement for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # Helper to track calls and enforce deadline
        calls = 0
        def timed_func(x):
            nonlocal calls
            if time.time() > deadline:
                return np.inf
            calls += 1
            return func(x)

        # Phase 1: Systematic Exploration using Sobol Sequence
        # 64 points provides better coverage than 20, while leaving budget for refinement
        n_sobol = 64
        try:
            sampler = Sobol(d=dim, scramble=True, seed=42)
            points = sampler.random(n_sobol)
        except Exception:
            # Fallback to random uniform if Sobol fails
            rng = np.random.default_rng(42)
            points = rng.uniform(0, 1, size=(n_sobol, dim))
            
        # Map from [0, 1]^d to [-5, 5]^d
        x_sobol = points * 10.0 - 5.0
        
        results = []
        for x in x_sobol:
            if time.time() > deadline:
                break
            fx = timed_func(x)
            results.append((fx, x))
        
        if not results:
            # Fallback if even initial phase fails
            x0 = np.zeros(dim)
            f0 = timed_func(x0)
            return x0, f0

        # Sort by fitness
        results.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start Nelder-Mead on the best candidates
        # We pick the top 3 distinct points to start local refinement
        remaining_budget = budget - calls
        n_starts = 3
        budget_per_start = max(10, remaining_budget // n_starts)
        
        best_x = results[0][1]
        best_f = results[0][0]
        
        for i in range(n_starts):
            if i >= len(results):
                break
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            start_x = results[i][1].copy()
            
            try:
                res = minimize(timed_func, start_x, method='Nelder-Mead', 
                               options={'maxiter': budget_per_start, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
