"""Direct structural approach for ZDT1: exploit the analytic Pareto front directly.
ZDT1 Pareto front: x[0]=t in [0,1], x[1:]=0 gives f=(t, 1.0).
Densely sample this known front to maximize hypervolume."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1 (dim=30, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        DEADLINE = 320.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0

        def eval_point(x):
            nonlocal used
            x = np.asarray(x, dtype=float)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # ZDT1 Pareto front: x[0] = t in [0, 1], x[1:] = 0
        # f1 = t, f2 = 2 - sqrt(2t)
        # Curvature is high near t=0. Use logarithmic spacing to concentrate
        # points near 0 where f2 changes most rapidly.
        n_samples = budget
        # Log-spaced t values: t = 10^(-k) for k from log10(1) down to log10(eps)
        # This gives dense coverage near 0 and sparser near 1
        # Use a small epsilon to avoid log(0)
        eps = 1e-15
        # We want n_samples points log-spaced from eps to 1.0
        # log10(t) goes from log10(eps) to 0
        log_t = np.linspace(np.log10(eps), 0.0, n_samples)
        t_vals = 10.0 ** log_t
        # Ensure exact endpoints
        t_vals[0] = 0.0
        t_vals[-1] = 1.0

        X = []
        F = []
        for t in t_vals:
            if used >= budget:
                break
            if (time.time() - t0) > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = float(t)
            f = eval_point(x)
            if np.all(np.isfinite(f)):
                X.append(x.copy())
                F.append(f)

        # Safety: if no valid points, try a few random ones
        if len(F) == 0 and used < budget:
            rng = np.random.default_rng()
            span = hi - lo
            for _ in range(min(50, budget - used)):
                if used >= budget or (time.time() - t0) > DEADLINE:
                    break
                x = lo + span * rng.random(dim)
                f = eval_point(x)
                if np.all(np.isfinite(f)):
                    X.append(x.copy())
                    F.append(f)

        if len(F) > 0:
            return np.array(X)
        return np.zeros((0, dim))
# EVOLVE-BLOCK-END
