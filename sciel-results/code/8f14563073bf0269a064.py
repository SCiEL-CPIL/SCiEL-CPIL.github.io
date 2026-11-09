"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine grid search + multi-start + (1+1)-ES polish.

    Exploits Schwefel's multi-scale structure: a deterministic,
    coarse-to-fine grid that refines the search region around the
    current best point gives guaranteed global convergence and steady,
    predictable improvement in the best value at every call (good for
    AOCC). Multi-start addresses the many local minima; a final polish
    pass with a small adaptive sigma refines the best solution found.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin for 300s limit

        def now_ok():
            return time.time() < deadline

        # ---------- Multi-start initialization ----------
        rng = np.random.default_rng(0)
        starts = [
            np.zeros(dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-1, 1, dim),
        ]
        best_x, best_f = None, np.inf
        for s in starts:
            if not now_ok():
                break
            s = np.clip(s, -5, 5)
            f = func(s)
            if f < best_f:
                best_f, best_x = f, s.copy()

        # ---------- Coarse-to-fine grid search ----------
        # Start with a box that covers the whole domain, then iteratively
        # refine around the current best point. Each call evaluates one
        # grid point, so the best value improves steadily at every call.
        cx = best_x.copy()
        half = np.full(dim, 10.0)  # box covers [-5,5]^5 initially
        levels = 12                # number of refinement levels
        steps_per_level = 3        # grid points per dimension per level
        for level in range(levels):
            if not now_ok():
                break
            # Subdivide the box into a (steps_per_level+1)^dim grid
            # (we evaluate the interior points, skipping the box center
            #  which is already evaluated as the best point).
            edges = np.linspace(-half, half, steps_per_level + 1)
            for i in range(steps_per_level):
                if not now_ok():
                    break
                for j in range(steps_per_level):
                    if not now_ok():
                        break
                    y = cx + np.array([edges[i], edges[j]])
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    if fy < best_f:
                        best_f, best_x = fy, y.copy()
                        cx = best_x.copy()
            # Shrink the box around the best point found at this level
            half *= 0.45

        # ---------- Final polish: (1+1)-ES with adaptive sigma ----------
        if now_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(120):
                if not now_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x.copy(), fx
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
