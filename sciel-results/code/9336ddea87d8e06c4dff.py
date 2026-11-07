"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start optimizer with L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        call_count = [0]

        def counted_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return val

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.clip(x, -5, 5).copy()

        # Phase 1: Diverse initial sampling using Sobol points
        # Use 50 points for better global coverage in 5D
        n_sobol = 50
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_sobol)
        sobol_pts = sobol_pts * 10.0 - 5.0

        # Add center point
        sobol_pts = np.vstack([sobol_pts, np.zeros(dim)])

        start_values = []
        for pt in sobol_pts:
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            try:
                fv = counted_func(pt)
            except Exception:
                fv = 1e10
            start_values.append((fv, pt.copy()))
            update_best(pt, fv)

        if not start_values:
            pt = np.zeros(dim)
            try:
                fv = counted_func(pt)
            except Exception:
                fv = 1e10
            start_values.append((fv, pt.copy()))
            update_best(pt, fv)

        # Sort by function value
        start_values.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using L-BFGS-B from top starts
        # We will try to refine the top 5 starts
        k = min(5, len(start_values))
        
        for i in range(k):
            if call_count[0] >= budget - 10:
                break
            if time.time() > deadline:
                break
            
            f0, x0 = start_values[i]
            
            # If this start is not better than current best, skip? 
            # No, it might be in a different basin. But if f0 > best_f, it's unlikely to be better.
            # However, for rugged functions, a slightly worse start might be in the global basin.
            # Let's try all top 5.
            
            remaining = budget - call_count[0]
            if remaining < 10:
                break
            
            # Allocate budget for this local search
            # L-BFGS-B is efficient, give it up to 100 iterations or 30% of remaining budget
            max_iter = max(10, min(100, remaining // 3))
            
            try:
                res = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_iter, 'maxls': 20}
                )
                
                # L-BFGS-B might not use all budget, but it will stop when converged
                # We need to update best with the result
                if np.isfinite(res.fun):
                    update_best(res.x, res.fun)
                else:
                    # If res.fun is inf, try the last point?
                    # minimize returns the best point found
                    if res.x is not None:
                         # We don't have the value, but we can assume it's not better than best_f
                         pass
                         
            except Exception:
                pass

        # Phase 3: Final polish with remaining budget
        if best_x is not None and call_count[0] < budget - 5:
            remaining = budget - call_count[0]
            if remaining > 0:
                try:
                    # One final L-BFGS-B run from the best point found
                    res = minimize(
                        counted_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': max(1, remaining), 'maxls': 20}
                    )
                    if np.isfinite(res.fun):
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # Fallback if nothing was evaluated
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                try:
                    best_f = counted_func(best_x)
                except Exception:
                    best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
