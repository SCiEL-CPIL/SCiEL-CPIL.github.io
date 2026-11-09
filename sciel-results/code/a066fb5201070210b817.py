"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Coarse-to-fine 1D grid search + Nelder-Mead polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Phase 1: Coordinate-wise 1D grid search to find global basin
        # Rastrigin has minima at integer points in each coordinate
        # We search each coordinate independently with a coarse grid
        
        # Start from origin (0,0,...,0) which is a natural starting point
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # Coarse grid search along each axis
        # Rastrigin period is 2π ≈ 6.28, so we need resolution < π/2 ≈ 1.57
        # Use step size ~1.0 for coarse phase
        coarse_step = 1.0
        fine_step = 0.1
        finer_step = 0.01
        
        # For each coordinate, do a 1D search
        for coord in range(dim):
            # Coarse search
            best_local = x.copy()
            best_local_f = fx
            for val in np.arange(-5.0, 5.0 + coarse_step, coarse_step):
                if time.time() > deadline:
                    break
                candidate = best_local.copy()
                candidate[coord] = val
                candidate = np.clip(candidate, -5, 5)
                f_val = func(candidate)
                calls += 1
                if f_val < best_local_f:
                    best_local_f = f_val
                    best_local = candidate
            x, fx = best_local, best_local_f

            # Fine search around the best
            center = x[coord]
            for val in np.arange(center - 1.0, center + 1.0 + fine_step, fine_step):
                if time.time() > deadline:
                    break
                candidate = x.copy()
                candidate[coord] = np.clip(val, -5, 5)
                f_val = func(candidate)
                calls += 1
                if f_val < fx:
                    fx = f_val
                    x = candidate

            # Finer search
            center = x[coord]
            for val in np.arange(center - 0.1, center + 0.1 + finer_step, finer_step):
                if time.time() > deadline:
                    break
                candidate = x.copy()
                candidate[coord] = np.clip(val, -5, 5)
                f_val = func(candidate)
                calls += 1
                if f_val < fx:
                    fx = f_val
                    x = candidate

        # Phase 2: Nelder-Mead polish from the refined starting point
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    func,
                    x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(budget - calls, 500),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < fx:
                    x, fx = result.x, result.fun
                    calls += result.nfev
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
