"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead optimization for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls_used = 0
        best_x = np.full(self.dim, np.nan)
        best_f = np.inf

        # Wrap func to count calls and enforce deadline
        def counted_func(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                raise RuntimeError("Budget exceeded")
            elapsed = time.time() - start_time
            if elapsed > deadline:
                raise TimeoutError("Time limit exceeded")
            calls_used += 1
            # Ensure x is a numpy array of floats
            x = np.asarray(x, dtype=float)
            return func(x)

        # 1. Global Initialization: Stratified sampling
        # Use 2^5 = 32 points for stratified, plus 18 random = 50 points
        n_stratified = 32
        edges = np.linspace(-5, 5, 2)  # 2 cells per dim -> 2^5 = 32
        # Actually linspace(-5, 5, 3) gives [-5, 0, 5]. 2 cells.
        # Let's use 2 cells per dimension.
        points = []
        for i in range(n_stratified):
            idx = np.unravel_index(i, (2,) * self.dim)
            p = np.zeros(self.dim)
            for d in range(self.dim):
                lo, hi = edges[idx[d]], edges[idx[d] + 1]
                p[d] = rng.uniform(lo, hi)
            points.append(p)
        
        # Add random points to fill up to 50
        while len(points) < 50:
            points.append(rng.uniform(-5, 5, self.dim))
        
        # Evaluate initial points
        for p in points:
            if calls_used >= self.budget:
                break
            try:
                f_val = counted_func(p)
                if f_val < best_f:
                    best_f = f_val
                    best_x = p.copy()
            except (RuntimeError, TimeoutError):
                break
            except Exception:
                continue

        # 2. Local Refinement: Nelder-Mead on the best few points
        # We have ~450 calls left.
        # Nelder-Mead uses dim+1 points per iteration.
        # Let's run it on the top 5 distinct starting points.
        
        # Sort initial points by function value
        # We need to store the values. Let's re-evaluate or keep track.
        # Since we didn't store all values, let's just pick the best_x found so far
        # and a few random perturbations of it, or just restart from best_x with different simplex sizes.
        
        # A better strategy: Keep a list of evaluated points and pick the top 5.
        # But we didn't store them all. Let's just do 10 restarts of Nelder-Mead
        # centered around the current best_x with varying initial simplex sizes.
        
        for i in range(10):
            if calls_used >= self.budget:
                break
            if time.time() - start_time > deadline:
                break
            
            # Create a starting point near best_x
            # If best_x is nan, use center
            if np.isnan(best_f):
                x0 = np.zeros(self.dim)
            else:
                # Perturb best_x slightly
                scale = 1.0 / (i + 1)
                x0 = np.clip(best_x + scale * rng.standard_normal(self.dim), -5, 5)
            
            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 50, # Limit iterations
                        'xatol': 1e-4,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except (RuntimeError, TimeoutError):
                break
            except Exception:
                continue

        return best_x, best_f
# EVOLVE-BLOCK-END
