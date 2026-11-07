"""Global-to-local optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, dual_annealing


class Optimizer:
    """Optimizer: global random sampling + local Nelder-Mead/CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(7)
        dim = self.dim
        lo, hi = -5.0, 5.0
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def record(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # Phase 1: Global exploration with 60 random points
        n_global = 60
        for i in range(n_global):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(lo, hi, dim)
            fx = func(x)
            record(x, fx)

        # Phase 2: Local refinement using multi-start Nelder-Mead
        # Use the top 5 points from Phase 1 as starting points
        # But since we don't store all, let's just use best and a few new random starts near best
        # Actually, let's do a more structured approach:
        # Use dual_annealing for a global search if time permits, otherwise local refinement.
        
        # Let's do local refinement from the best point found in Phase 1
        x0 = best_x.copy()
        
        # Try local optimization from multiple nearby points
        n_starts = 5
        starts = [x0]
        for i in range(n_starts - 1):
            if calls_made >= budget or time.time() > deadline:
                break
            # Perturb best point
            x_try = x0 + rng.normal(0, 1.0, dim)
            x_try = np.clip(x_try, lo, hi)
            fx_try = func(x_try)
            record(x_try, fx_try)
            if fx_try < best_f + 10:  # Only use if reasonably close
                starts.append(x_try)
        
        # Local refinement using Nelder-Mead
        for x_start in starts:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Define a counter to limit evaluations per local run
            remaining = budget - calls_made
            max_iter_local = min(200, max(50, remaining // len(starts)))
            
            try:
                result = minimize(
                    func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_local,
                        'maxfev': min(150, budget - calls_made),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                # The result contains the best point and value
                if result.success or result.fun < best_f:
                    record(result.x, result.fun)
            except Exception:
                pass

        # Phase 3: If time and budget remain, do a final global search with dual_annealing
        if calls_made < budget and time.time() < deadline - 30:
            try:
                remaining = budget - calls_made
                if remaining > 100:
                    result_da = dual_annealing(
                        func,
                        bounds=[(lo, hi)] * dim,
                        initial_temp=5230.0,
                        restart_temp_ratio=2e-5,
                        visit=2.6,
                        seed=int(rng.integers(0, 2**31)),
                        maxiter=100,
                        maxfev=remaining
                    )
                    record(result_da.x, result_da.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
