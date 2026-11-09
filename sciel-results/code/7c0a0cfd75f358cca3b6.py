"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Two-phase optimizer: Sobol exploration + Multi-start Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0
        # Store top candidates from exploration for multi-start local search
        top_candidates = []  # List of (fx, x)

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            # Maintain a list of top 5 candidates
            top_candidates.append((fx, x.copy()))
            top_candidates.sort(key=lambda t: t[0])
            if len(top_candidates) > 5:
                top_candidates.pop()

        def safe_eval(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            try:
                fx = func(x)
                calls_used += 1
                return fx
            except Exception:
                calls_used += 1
                return np.inf

        # Phase 1: Coarse search using Sobol sequence
        # We use a portion of the budget for global exploration
        # Using ~20% of budget for exploration seems reasonable for dim=5
        n_explore = min(100, budget // 5)
        
        # Generate Sobol points
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            sobol_points = sampler.random(n_explore)
        except Exception:
            # Fallback to random if Sobol fails
            sobol_points = rng.random((n_explore, dim))

        # Map from [0, 1]^dim to [-5, 5]^dim
        # The problem states minimizer lies in [-4, 4]^5, so we could restrict,
        # but Sobol over [-5, 5] is safe. Let's map to [-4.5, 4.5] to be slightly more focused?
        # No, stick to [-5, 5] to be safe, or maybe [-4.5, 4.5] since min is in [-4,4].
        # Let's use [-4.5, 4.5] to increase density in the likely region.
        sobol_points = sobol_points * 9.0 - 4.5

        # Evaluate all Sobol points
        for x in sobol_points:
            if time.time() > deadline or calls_used >= budget:
                break
            fx = safe_eval(x)
            if np.isfinite(fx):
                update_best(x, fx)

        # If no points were evaluated (shouldn't happen), fallback to random
        if best_x is None:
            x = np.zeros(dim)
            fx = safe_eval(x)
            update_best(x, fx)
            if best_x is None:
                # Absolute fallback
                best_x = np.zeros(dim)
                best_f = 0.0

        # Phase 2: Local refinement using Nelder-Mead
        # We will try to refine the best point found so far.
        # To be more robust, we will do multi-start from the top few candidates.
        
        remaining_budget = budget - calls_used
        if remaining_budget <= 10:
            return best_x, best_f

        # Define a local search function that respects global budget
        def local_search(x0, max_evals):
            nonlocal calls_used, best_x, best_f
            
            if calls_used >= budget or time.time() > deadline:
                return
                
            # Wrap func for NM to respect budget and update global best
            def nm_func(x):
                nonlocal calls_used
                if calls_used >= budget or time.time() > deadline:
                    return np.inf
                try:
                    fx = func(x)
                    calls_used += 1
                    update_best(x, fx)
                    return fx
                except Exception:
                    calls_used += 1
                    return np.inf

            # Nelder-Mead options
            # Use adaptive NM if available in SciPy version, otherwise standard
            options = {
                'xatol': 1e-6, 
                'fatol': 1e-6, 
                'maxiter': max_evals // 2, 
                'maxfev': max_evals
            }
            try:
                # Check if adaptive is supported (SciPy >= 1.10 or so)
                # For safety, stick to standard options or try/except
                res = minimize(nm_func, x0, method='Nelder-Mead', options=options)
            except Exception:
                try:
                    # Fallback to simpler options
                    res = minimize(nm_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': max_evals // 2, 'maxfev': max_evals})
                except Exception:
                    pass

        # Multi-start strategy:
        # 1. Run NM from the global best point.
        # 2. If budget remains, run NM from the 2nd and 3rd best points.
        
        # Sort top_candidates to ensure we have the best ones
        top_candidates.sort(key=lambda t: t[0])
        
        # Run local search on top candidates
        # Allocate budget roughly equally among top 3 candidates
        # Or just run sequentially until budget is exhausted.
        
        # Let's run on the best candidate first with a large chunk
        # Then on the next best with remaining budget
        
        # We have top_candidates list of (fx, x)
        # We already evaluated these in Phase 1, so we don't need to re-evaluate the start point
        # but NM will evaluate neighbors.
        
        for i, (fx, x) in enumerate(top_candidates[:3]):
            if calls_used >= budget or time.time() > deadline:
                break
            
            # Allocate remaining budget to this search
            # If it's the first, give it 60% of remaining, next 30%, next 10%?
            # Or just let it run until budget is low.
            # Since NM stops when maxfev is reached, we can just set maxfev to remaining_budget.
            
            # To prevent one search from eating all budget and leaving none for others,
            # we can cap the first one.
            if i == 0:
                max_evals = remaining_budget
            else:
                # Use remaining budget
                max_evals = budget - calls_used
                
            if max_evals <= 10:
                break
                
            local_search(x, max_evals)

        return best_x, best_f
# EVOLVE-BLOCK-END
