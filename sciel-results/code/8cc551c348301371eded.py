"""Rewritten program using SQP (SLSQP) with multi-start initialization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol


class Optimizer:
    """Multi-start SLSQP optimizer for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return func(x)

        # Phase 1: Sobol sampling for better coverage
        # Use about 20-30% of budget for initialization
        n_samples = min(30, max(10, budget // 5))
        
        # Generate Sobol points
        sobol = Sobol(d=dim, seed=42, scramble=True)
        # We need to generate enough points. Sobol requires power of 2.
        n_pow2 = 1
        while n_pow2 < n_samples:
            n_pow2 *= 2
        points = sobol.random(n_pow2)
        # Map [0, 1] to [-5, 5]
        points = points * 10.0 - 5.0
        
        candidates = []
        for i in range(n_samples):
            if call_count[0] >= budget - 50: # Keep reserve for optimization
                break
            x0 = points[i]
            f0 = wrapped_func(x0)
            candidates.append((f0, x0.copy()))
            
        if not candidates:
            # Fallback if something went wrong or budget was tiny
            x0 = np.zeros(dim)
            f0 = wrapped_func(x0)
            candidates.append((f0, x0))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Multi-start SLSQP from top candidates
        # We will allocate remaining budget to these runs.
        # Let's try the top 3 candidates.
        n_starts = min(3, len(candidates))
        
        # Estimate max iterations for each start based on remaining budget
        # Each iteration ~ dim + 1 = 6 calls
        cost_per_iter = dim + 1
        
        for i in range(n_starts):
            f_start, x_start = candidates[i]
            if f_start < best_f:
                best_f = f_start
                best_x = x_start.copy()
            
            # Determine how many iterations to allow for this start
            # Give more iterations to the first (best) start
            if i == 0:
                alloc_frac = 0.5
            elif i == 1:
                alloc_frac = 0.3
            else:
                alloc_frac = 0.2
                
            remaining_budget = budget - call_count[0] - 20 # Reserve 20 for final polish
            if remaining_budget <= 0:
                break
                
            max_iter = int((remaining_budget * alloc_frac) / cost_per_iter)
            max_iter = max(5, min(max_iter, 100)) # Cap at 100 to prevent runaway
            
            try:
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final high-precision SLSQP from the best point found
        if best_x is not None and call_count[0] < budget - 5:
            remaining = budget - call_count[0] - 5
            if remaining > cost_per_iter:
                max_iter_final = int(remaining / cost_per_iter)
                max_iter_final = max(5, min(max_iter_final, 200))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter_final, 'ftol': 1e-15, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Ensure we return the best solution found
        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                best_f = func(best_x)
            else:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
