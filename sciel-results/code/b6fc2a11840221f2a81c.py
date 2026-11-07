"""Rewritten program using SQP (SLSQP) with multi-start initialization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start SLSQP optimizer for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Bounds for SLSQP
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf

        # Track total function calls made
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return float(func(x))

        def lhs_sample(n, dim, rng):
            """Generate Latin Hypercube samples in [-5, 5]^dim."""
            samples = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                samples[:, d] = (perm + rng.random(n)) / n
                samples[:, d] = samples[:, d] * 10.0 - 5.0
            return samples

        # Phase 1: Latin Hypercube sampling for better space coverage
        n_lhs = min(20, max(8, budget // 30))
        lhs_points = lhs_sample(n_lhs, dim, rng)
        
        for i in range(n_lhs):
            if call_count[0] >= budget - 50:
                break
            x0 = lhs_points[i]
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: Multi-start SLSQP with adaptive perturbation scales
        if best_x is not None:
            # First: SLSQP from the best point
            remaining = budget - call_count[0] - 30
            if remaining > 15:
                max_iter = max(15, min(60, remaining // (dim + 2)))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

            # Second: Try multiple perturbation scales around best point
            # Use geometrically decreasing scales for diversity
            n_perturb = min(10, max(4, (budget - call_count[0] - 30) // 40))
            scales = np.logspace(0, -2, n_perturb)  # from 1.0 down to 0.01
            
            for j in range(n_perturb):
                if call_count[0] >= budget - 25:
                    break
                
                scale = scales[j]
                # Generate a random direction
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                
                x0 = best_x + scale * direction
                x0 = np.clip(x0, -5.0, 5.0)
                
                f0 = wrapped_func(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

                # Run SLSQP from this point if we have budget
                remaining = budget - call_count[0] - 15
                if remaining > 10:
                    max_iter = max(8, min(35, remaining // (dim + 2)))
                    try:
                        res = minimize(
                            wrapped_func,
                            x0,
                            method='SLSQP',
                            bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        # Phase 3: Nelder-Mead fallback for final polishing
        if best_x is not None and call_count[0] < budget - 10:
            remaining = budget - call_count[0] - 5
            if remaining > 8:
                max_iter_nm = max(20, min(80, remaining // 2))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': max_iter_nm, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 4: Final high-precision SLSQP
        if best_x is not None and call_count[0] < budget - 5:
            remaining = budget - call_count[0] - 3
            if remaining > 5:
                max_iter_final = max(15, min(100, remaining // (dim + 1)))
                try:
                    res = minimize(
                        wrapped_func,
                        best_x,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter_final, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Ensure we return the best solution found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
