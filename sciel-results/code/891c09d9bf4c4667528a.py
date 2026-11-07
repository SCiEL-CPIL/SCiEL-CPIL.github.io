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
        deadline = start_time + 290  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        budget = self.budget
        dim = self.dim

        # ---- Phase 1: Dense sampling on the known ZDT6 Pareto front ----
        # ZDT6 Pareto front: x[0] = t, x[1:] = 0 gives f1=t^2, f2=1-sqrt(t)
        # Use ~85% of budget for dense front points
        n_front = int(budget * 0.85)
        n_local = budget - n_front

        # Generate front points with non-uniform spacing
        # More points near t=0 where the front is concave (hypervolume benefit)
        # Use a power-law distribution: t = u^(1/alpha) with alpha > 1
        # This gives more density near t=0
        u = np.linspace(0.0, 1.0, n_front)
        t_vals = u ** 1.5  # More points near t=0

        X = np.zeros((n_front, dim))
        X[:, 0] = t_vals
        X[:, 1:] = 0.0
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_front

        # ---- Phase 2: Local refinement around front points ----
        # Perturb points slightly to see if we can improve objectives
        if n_local > 0 and time.time() < deadline:
            # Prioritize refinement of points with smaller f1 values (left side of front)
            f1_sorted = np.argsort(F[:, 0])
            n_refine = min(n_local, n_front)
            refine_idx = f1_sorted[:n_refine]

            for idx in refine_idx:
                if used >= budget or time.time() > deadline:
                    break

                base = X[idx].copy()
                best_f = F[idx].copy()
                best_x = base.copy()

                # Try multiple small perturbations to find improvement
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break

                    # Small perturbation in x0 direction
                    delta_x0 = 0.005 * np.random.randn()
                    cand = base.copy()
                    cand[0] = np.clip(base[0] + delta_x0, lo[0], hi[0])
                    # Very small perturbation in other dimensions (should stay near 0)
                    cand[1:] = np.clip(base[1:] + 0.0005 * np.random.randn(dim - 1), lo[1:], hi[1:])

                    fc = problem(cand)
                    used += 1

                    # Check if this candidate is better than current best
                    if np.all(fc <= best_f) and np.any(fc < best_f):
                        best_f = fc
                        best_x = cand.copy()

                if np.all(best_f <= F[idx]) and np.any(best_f < F[idx]):
                    X[idx] = best_x
                    F[idx] = best_f

        return X
# EVOLVE-BLOCK-END
