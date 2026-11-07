"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start L-BFGS-B with derivative-free fallback for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = [0]
        
        def wrapped_func(x):
            calls[0] += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # 1. Global Exploration using Sobol Sequence for better coverage
        n_global = 60
        if n_global > budget // 2:
            n_global = max(10, budget // 10)

        try:
            # Use Sobol sequence for low-discrepancy sampling
            sobol = qmc.Sobol(d=dim, scramble=True, seed=123)
            points_unit = sobol.random(n_global)
            points = points_unit * 10.0 - 5.0
        except Exception:
            # Fallback to Halton or Random if Sobol fails
            try:
                halton = qmc.Halton(d=dim, scramble=True, seed=123)
                points_unit = halton.random(n_global)
                points = points_unit * 10.0 - 5.0
            except Exception:
                rng = np.random.default_rng(123)
                points = rng.uniform(-5, 5, (n_global, dim))

        best_x = None
        best_f = np.inf
        candidates = []

        # Evaluate global points
        for p in points:
            if calls[0] >= budget - 10:
                break
            fp = wrapped_func(p)
            candidates.append((fp, p.copy()))
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Sort candidates by function value
        candidates.sort(key=lambda x: x[0])

        # 2. Local Refinement with L-BFGS-B on the best candidates
        k_local = 4
        if len(candidates) < k_local:
            k_local = len(candidates)

        remaining_budget = budget - calls[0]
        # Reserve 50 calls for final polish
        budget_per_run = max(10, (remaining_budget - 50) // k_local) if k_local > 0 else 0

        for i in range(k_local):
            if calls[0] >= budget - 20:
                break
            f0, x0 = candidates[i]
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': budget_per_run,
                        'gtol': 1e-8,
                        'ftol': 1e-12
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass

        # 3. Final Polish
        if calls[0] < budget - 5 and best_x is not None:
            remaining = budget - calls[0]
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': remaining - 5,
                        'gtol': 1e-10,
                        'ftol': 1e-14
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0)
            except Exception:
                pass
        
        # Ensure we made at least one call
        if calls[0] == 0:
            x0 = np.zeros(dim)
            best_f = wrapped_func(x0)
            best_x = x0.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
