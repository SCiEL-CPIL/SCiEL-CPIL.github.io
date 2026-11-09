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
        init_points = []
        
        # 1. Exact zero
        init_points.append(np.zeros(dim))
        
        # 2. Gaussian sampling centered at 0 (sigma=1.0)
        # 24 points
        for _ in range(24):
            p = rng.normal(0, 1.0, dim)
            init_points.append(np.clip(p, -5, 5))
            
        # 3. Uniform sampling for exploration
        # 25 points
        for _ in range(25):
            p = rng.uniform(-4, 4, dim)
            init_points.append(p)
        
        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)
        
        # Sort indices by function value (ascending)
        sorted_indices = np.argsort(init_values)
        
        # 2. Funnel Local Search (Nelder-Mead)
        
        # Phase 1: Very short NM on all 50 points
        # maxiter=2 is very cheap, allows quick filtering
        if calls < self.budget - 200:
            for seed_idx in sorted_indices:
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 2, 
                                       'xatol': 1e-4, 
                                       'fatol': 1e-6, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 2: Medium NM on top 10 points
        if calls < self.budget - 100:
            top_indices = sorted_indices[:10]
            for seed_idx in top_indices:
                if time.time() - start_time > time_limit - 10.0:
                    break
                seed = init_points[seed_idx].copy()
                # Note: We should restart from the current best of that basin if we tracked it, 
                # but for simplicity we restart from the initial point or the best found so far?
                # Restarting from the initial point might miss the progress from Phase 1.
                # It's better to restart from the best point found in Phase 1 for that seed?
                # We didn't track per-seed bests.
                # Let's just use the global best_x as the seed for the next phase? 
                # No, that kills diversity.
                # Let's use the initial points again, but NM will likely converge to the same local min.
                # To be safe, we can just run NM on the initial points again, but with more iterations.
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead', 
                                   options={
                                       'maxiter': 10, 
                                       'xatol': 1e-5, 
                                       'fatol': 1e-8, 
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Phase 3: Long NM on top 1 point
        if calls < self.budget - 20 and best_x is not None:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={
                                   'maxiter': 50, 
                                   'xatol': 1e-7, 
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
