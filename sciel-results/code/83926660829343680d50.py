"""Lattice basin-locator + local Nelder-Mead refiner for Rastrigin f3."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Lattice-based global basin search followed by local Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 250.0  # safety margin well under 300 s
        calls = 0

        def F(x):
            nonlocal calls
            calls += 1
            return float(func(np.asarray(x, dtype=float)))

        # ---- Phase 1: coarse lattice scan over integer grid in [-5, 5]^dim ----
        # Rastrigin has peaks at integer coordinates, so the global minimum sits
        # near an integer point. A full integer grid gives 11^dim candidates.
        npts = 0
        best_x, best_f = None, np.inf
        grid = np.arange(-5, 6, 1.0)
        for idx in np.ndindex(*([len(grid)] * dim)):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
            pt = np.array([grid[i] for i in idx], dtype=float)
            f = F(pt)
            npts += 1
            if f < best_f:
                best_f, best_x = f, pt.copy()

        # ---- Phase 2: local Nelder-Mead refinement from the best basin ----
        if best_x is not None and calls < self.budget:
            try:
                res = minimize(
                    F, best_x, method="Nelder-Mead",
                    options={"maxiter": 6000, "maxfev": self.budget - calls,
                             "xatol": 1e-8, "fatol": 1e-10},
                )
                if res.fun < best_f:
                    best_f, best_x = float(res.fun), np.array(res.x)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
