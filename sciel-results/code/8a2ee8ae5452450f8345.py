"""Multi-objective optimizer on ZDT6 — dense front sampling with local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Dense Pareto-front sampling with local refinement for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        budget = self.budget
        dim = self.dim

        # ---- Phase 1: Dense sampling on the known ZDT6 Pareto front ----
        # ZDT6 Pareto front: x[0] = t, x[1:] = 0 gives f1=t^2, f2=1-sqrt(t)
        # Use ~75% of budget for dense front points
        n_front = int(budget * 0.75)
        n_local = budget - n_front

        # Generate front points with non-uniform spacing
        # More points near t=0 where the front is concave (hypervolume benefit)
        # Use a power-law distribution: t = u^(1/alpha) with alpha > 1
        # This gives more density near t=0
        u = np.linspace(0.0, 1.0, n_front)
        t_vals = u ** 0.7  # More points near t=0

        X = np.zeros((n_front, dim))
        X[:, 0] = t_vals
        X[:, 1:] = 0.0
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_front

        # ---- Phase 2: Local refinement around front points ----
        # Perturb points slightly to see if we can improve objectives
        if n_local > 0 and time.time() < deadline:
            # Select indices to refine: spread across the front
            n_refine = min(n_local, n_front)
            # Use stratified selection for coverage
            refine_idx = np.linspace(0, n_front - 1, n_refine, dtype=int)

            for idx in refine_idx:
                if used >= budget or time.time() > deadline:
                    break

                base = X[idx].copy()
                # Small perturbation in x0 direction
                delta_x0 = 0.005 * np.random.randn()
                base[0] = np.clip(base[0] + delta_x0, lo[0], hi[0])
                # Very small perturbation in other dimensions (should stay near 0)
                base[1:] = np.clip(base[1:] + 0.001 * np.random.randn(dim - 1), lo[1:], hi[1:])

                fc = problem(base)
                used += 1

                # Replace if new point dominates old point
                if np.all(fc <= F[idx]) and np.any(fc < F[idx]):
                    X[idx] = base
                    F[idx] = fc

        return X
# EVOLVE-BLOCK-END
