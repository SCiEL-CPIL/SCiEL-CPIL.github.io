"""Rewritten program using a tiered local search strategy with improved initial sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Katsuura function (f23).
    Strategy:
    1. Broad Initial Sampling: 50 points to cover the space well.
    2. Funnel Local Search: 
       - Phase 1: Very short Nelder-Mead on all 50 points to filter out bad basins.
       - Phase 2: Medium Nelder-Mead on top 10 points.
       - Phase 3: Long Nelder-Mead on top 1 point.
    This maximizes exploration early and exploits the best basin later.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # 1. Initial Diverse Sampling
        # Strategy: Focus on the origin (likely global min for Katsuura) and local neighborhood,
        # with some global exploration.
        init_points = []
        
        # 1. Exact zero (Highest priority)
        init_points.append(np.zeros(dim))
        
        # 2. Local Gaussian sampling around origin (sigma=0.5, 1.0)
        # Katsuura is oscillatory, so local minima are frequent.
        # 10 points with sigma=0.5
        for _ in range(10):
            p = rng.normal(0, 0.5, dim)
            init_points.append(np.clip(p, -5, 5))
            
        # 10 points with sigma=1.0
        for _ in range(10):
            p = rng.normal(0, 1.0, dim)
            init_points.append(np.clip(p, -5, 5))
            
        # 3. Uniform sampling for global exploration
        # 10 points
        for _ in range(10):
            p = rng.uniform(-5, 5, dim)
            init_points.append(p)
        
        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort indices by function value (ascending)
        # We want to process the best candidates first.
        sorted_indices = np.argsort(init_values)
        
        # 2. Prioritized Local Search
        # Instead of sweeping all points with shallow searches, we dive deep into the best ones first.
        # This is crucial for AOCC because early progress on the best basin dominates the score.
        
        # Tier 1: Deep search on the top 3 candidates
        # Allocate significant budget here.
        # If the global min is in the top 3 (highly likely if origin is included), we converge fast.
        top_k = 3
        if calls < self.budget - 50:
            for i in range(min(top_k, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    # Medium-Long NM to ensure convergence in the basin
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 50, 
                                       'xatol': 1e-6, 
                                       'fatol': 1e-8, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 2: Medium search on the next 5 candidates (4-8)
        if calls < self.budget - 50:
            for i in range(top_k, min(top_k + 5, len(sorted_indices))):
                seed_idx = sorted_indices[i]
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 20, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-7, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Tier 3: Polish the global best found
        # Use the remaining budget to refine the best point found so far.
        if calls < self.budget - 5 and best_x is not None:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={
                                   'maxiter': 100, 
                                   'xatol': 1e-8, 
                                   'fatol': 1e-10, 
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
