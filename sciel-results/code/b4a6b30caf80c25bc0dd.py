"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """LHS global exploration + Nelder-Mead local refinement with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for global exploration
        n_lhs = min(200, budget // 3)
        n_lhs = max(n_lhs, 30)

        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.default_rng(seed=42 + d).permutation(n_lhs)
            lhs_points[:, d] = (perm + np.random.default_rng(seed=100 + d).uniform(size=n_lhs)) / n_lhs
        lhs_points = lhs_points * 10.0 - 5.0  # map to [-5, 5]

        # Evaluate LHS points
        evals = []
        best_x = None
        best_f = np.inf
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            evals.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        used = n_lhs

        # Phase 2: Select top-k candidates and refine with Nelder-Mead
        evals.sort(key=lambda t: t[0])
        n_refine = min(5, len(evals))
        remaining = budget - used

        if remaining <= 0:
            return best_x, best_f

        # Allocate remaining budget among top-k refinements
        budget_per_refine = max(remaining // n_refine, 5)

        for i in range(n_refine):
            if time.time() > deadline:
                break
            if remaining <= 0:
                break

            x0 = evals[i][1].copy()
            f0 = evals[i][0]

            # Nelder-Mead refinement
            call_count = [0]
            def nm_func(x):
                call_count[0] += 1
                if call_count[0] > budget_per_refine:
                    return f0  # stop
                return func(x)

            try:
                res = minimize(nm_func, x0, method='Nelder-Mead',
                               options={'maxiter': budget_per_refine,
                                        'xatol': 1e-4, 'fatol': 1e-4,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            used += call_count[0]
            remaining = budget - used

        # Phase 3: If budget remains, do a second round of LHS around best region
        if remaining > 10 and time.time() < deadline:
            n_extra = min(remaining - 5, 100)
            for i in range(n_extra):
                if time.time() > deadline:
                    break
                if remaining <= 5:
                    break
                # Sample around current best
                x = best_x + np.random.default_rng(seed=200 + i).normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                remaining -= 1

        return best_x, best_f
# EVOLVE-BLOCK-END
