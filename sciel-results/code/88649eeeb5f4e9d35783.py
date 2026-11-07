# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import cma


class Optimizer:
    """Global CMA-ES + cheap local Nelder-Mead polish for the Rastrigin (f3) task."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.monotonic()
        deadline = start + 300.0
        margin = 8.0
        rng = np.random.default_rng(0)

        def elapsed_ok():
            return (time.monotonic() + margin) < deadline

        # ---- global exploration via CMA-ES (robust on multimodal Rastrigin) ----
        x0 = rng.uniform(-4.0, 4.0, dim)
        bounds = [(-5.0, 5.0)] * dim
        opts = {
            "maxiter": 5000,
            "bounds": bounds,
            "seed": 0,
            "tolfun": 1e-12,
            "verbose": -9,
            "save_history": False,
            "maxfevals": budget,
            "tolx": 1e-9,
            "tolfunint": 1e-9,
            "CMA_adaptation_of_maxiter": True,
        }
        try:
            res = cma.fmin(func, x0, sigma0=1.6, options=opts)
            if res is None:
                res = x0
            best_x = np.asarray(res, dtype=float)
            best_f = float(func(best_x))
        except Exception:
            best_x = x0
            best_f = float(func(best_x))

        # ---- local polish: cheap damped Nelder-Mead, only while a gap remains ----
        # (guard ensures we stop safely before the deadline)
        for _ in range(4):
            if not elapsed_ok():
                break
            try:
                loc = minimize(
                    func,
                    best_x,
                    method="Nelder-Mead",
                    options={"maxiter": 250, "maxfev": 250, "xatol": 1e-7, "fatol": 1e-10},
                )
                if loc.x is not None:
                    best_x = np.asarray(loc.x, dtype=float)
                    best_f = float(func(best_x))
            except Exception:
                break
            if best_f < 1e-3:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
