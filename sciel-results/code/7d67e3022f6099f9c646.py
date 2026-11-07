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
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Wrapper to track calls and enforce deadline
        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(x))
                if val < best_f:
                    best_f = val
                    best_x = np.array(x, dtype=float)
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(12345)
        
        # Phase 1: Global Exploration
        # f22 is highly multi-modal. We need to cover the space well.
        # 150 random points seems like a reasonable trade-off for 500 budget.
        # We can also use a stratified sampling approach (Sobol-like) but random is simpler and robust.
        n_explore = 150
        candidates = []
        
        # Add center and corners to ensure coverage
        base_points = [
            np.zeros(dim),
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
        ]
        
        for x in base_points:
            f_val = eval_and_track(x)
            candidates.append((f_val, x))
            
        for i in range(n_explore - len(base_points)):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Refinement
        # Pick top K candidates. 
        # L-BFGS-B is efficient but can get stuck in local minima.
        # We allocate remaining budget among top candidates.
        # If we have ~350 calls left, we can afford ~5-10 local runs.
        # Each L-BFGS-B run on 5D typically takes < 50 evals to converge or hit maxiter.
        
        # Dynamic K: if we have lots of budget left, do more restarts.
        remaining_budget = budget - calls
        if remaining_budget < 20:
            return best_x, best_f
            
        # Estimate max local runs
        # Assume ~30 evals per local run on average (conservative)
        max_local_runs = min(15, max(1, int(remaining_budget / 30)))
        
        # We want to diversify the local starts.
        # Instead of just top 5, maybe pick top 5, then next best from a different region?
        # For simplicity, stick to top K but ensure we don't pick duplicates or very close points.
        # A simple distance check can help.
        
        selected_starts = []
        min_dist = 1.0  # Minimum distance between selected starts
        
        for f_val, x in candidates:
            if len(selected_starts) >= max_local_runs:
                break
            # Check distance to already selected
            is_too_close = False
            for _, x_sel in selected_starts:
                if np.linalg.norm(x - x_sel) < min_dist:
                    is_too_close = True
                    break
            if not is_too_close:
                selected_starts.append((f_val, x))
                
        # If we didn't fill up max_local_runs due to distance constraint, just take next best
        if len(selected_starts) < max_local_runs:
            for f_val, x in candidates:
                if len(selected_starts) >= max_local_runs:
                    break
                if (f_val, x) not in selected_starts:
                    selected_starts.append((f_val, x))

        # Execute local optimizations
        for f_val, x0 in selected_starts:
            if calls >= budget or time.time() > deadline:
                break
            
            # If budget is very tight, skip
            if budget - calls < 5:
                break
                
            try:
                # L-BFGS-B with bounds
                # maxiter is in iterations, not function calls. 
                # For 5D, convergence is fast. 100 iter is plenty.
                # We rely on the wrapper to stop if budget/time runs out.
                res = minimize(
                    eval_and_track, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 20}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
