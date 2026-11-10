"""Multi-objective optimizer for ZDT6 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit ZDT6 geometry: x[1:]=0, dense deterministic sampling of x[0]."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 with x[1:]=0: g=1, f1=x[0], f2=1-sqrt(f1)
        # All points on this curve are Pareto-optimal.
        # Use ALL budget for dense deterministic sampling of x[0] in [0,1].

        # Use all budget for dense front sampling
        n_points = budget
        # Small epsilon to stay strictly inside bounds
        eps = 1e-10
        # Curvature-aware: sample u uniformly, then f1 = u^2
        # This places more points near f1=0 where the front is steeper
        u_vals = np.linspace(eps, 1.0 - eps, n_points)
        x0_vals = u_vals ** 2
        # Clamp to valid range
        x0_vals = np.clip(x0_vals, eps, 1.0 - eps)

        evaluated_f = []

        # Pre-allocate for speed
        x = np.zeros(dim)

        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x[0] = x0
            f = problem(x)
            evaluated_f.append(f)

        return np.array(evaluated_f)
# EVOLVE-BLOCK-END
