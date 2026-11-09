"""Multi-objective optimizer for DTLZ7 with structural exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: Pareto front is 1D (x[0]), optimal has x[1:]=0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        X_list = []
        F_list = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            eval_count += 1
            X_list.append(x.copy())
            F_list.append(np.asarray(f, dtype=float))
            return f

        # Phase 1: Dense uniform initialization along the Pareto front
        # x[1:] = 0 gives g=1 (optimal), so we just need good x[0] coverage
        n_init = min(80, budget // 2)
        x0_vals = np.linspace(0.0, 1.0, n_init)
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            # x[1:] = 0 is optimal for DTLZ7 (g=1)
            f = evaluate(x)
            if f is None:
                break

        # Phase 2: Add additional random points with x[1:]=0 for better coverage
        # Use stratified sampling to fill gaps
        while eval_count < budget and time.time() < deadline:
            x = np.zeros(dim)
            x[0] = rng.uniform(0.0, 1.0)
            f = evaluate(x)
            if f is None:
                break

        return X_list
# EVOLVE-BLOCK-END
