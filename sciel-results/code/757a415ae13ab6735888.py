"""Optimizer for BBOB-style f15 (Rastrigin rotated) on [-5, 5]^5.

Strategy tuned for AOCC (early progress dominates):
  * Front-load cheap, high-impact exploration (a coarse global grid + a few
    cheap local polishes) so the best-so-far value improves quickly.
  * CMA-ES as the main robust black-box workhorse (handles the rotation),
    restarted from the best point found so far so progress is monotone.
  * L-BFGS-B polish on the smooth local model to convert a good point into a
    great one quickly.
  * Explicit wall-clock deadline guard with a safety margin.
"""
import time
import numpy as np
from scipy.optimize import minimize as _sp_min


class Optimizer:
    """Continuous black-box minimizer for f15 (Rastrigin rotated) on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0          # 330 s hard kill -> 50 s safety margin
        dim = self.dim
        budget = self.budget

        def _left():
            return time.time() < deadline

        # ---- state ----
        x_best = np.zeros(dim)
        f_best = np.inf
        calls = 0

        def _wrap(x):
            nonlocal calls
            calls += 1
            return func(x)

        def _update(x, f):
            nonlocal x_best, f_best
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # ---- 1) cheap global grid: front-load early progress ----
        # 3^5 = 243 points, step 2.5. For Rastrigin with shift in [-4,4], a
        # coarse grid reliably lands near a local minimum, giving a strong
        # early b_i and a solid seed for CMA-ES.
        grid_levels = 3
        gvals = np.linspace(-5.0, 5.0, grid_levels)
        for i0 in range(grid_levels):
            for i1 in range(grid_levels):
                for i2 in range(grid_levels):
                    for i3 in range(grid_levels):
                        for i4 in range(grid_levels):
                            if not _left():
                                break
                            x = np.array([gvals[i0], gvals[i1], gvals[i2],
                                          gvals[i3], gvals[i4]], dtype=float)
                            _update(x, _wrap(x))
                        if not _left():
                            break
                    if not _left():
                        break
                if not _left():
                    break
            if not _left():
                break
        if not _left():
            return x_best, f_best

        # ---- 2) cheap local polish on the best grid point ----
        try:
            res = _sp_min(_wrap, x_best, method="L-BFGS-B",
                          bounds=[(-5.0, 5.0)] * dim,
                          options={"maxiter": 25, "maxls": 12})
            if res.x is not None:
                _update(res.x, float(res.fun))
        except Exception:
            pass
        if not _left():
            return x_best, f_best

        # ---- 3) CMA-ES main loop (restarted from the current best) ----
        # Each restart uses a small budget and seeds from the global best, so
        # the best-so-far value is monotone non-increasing across restarts.
        max_sigma = 4.0
        restart_budgets = [100, 70, 50, 40, 30]
        for rb in restart_budgets:
            if not _left():
                break
            x0 = x_best.copy()
            sigma0 = min(2.5, max_sigma)
            pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))
            try:
                from cma import CMA, CMAOptions
            except Exception:
                break
            try:
                opts = CMAOptions(
                    popsize=pop,
                    maxfevals=rb,
                    seed=12345,
                    bounds=[(-5.0, 5.0)] * dim,
                    maxfevals_noimprove=30,
                    tolfun=1e-12,
                    tolx=1e-8,
                )
                es = CMA(x0, sigma0, opts)
                while not es.stop() and _left():
                    es.ask()
                    es.tell()
                # record best from this run
                b = es.best
                if b.x is not None:
                    _update(b.x, float(b.f))
            except Exception:
                break
        if not _left():
            return x_best, f_best

        # ---- 4) final polish with remaining budget ----
        try:
            res = _sp_min(_wrap, x_best, method="L-BFGS-B",
                          bounds=[(-5.0, 5.0)] * dim,
                          options={"maxiter": 40, "maxls": 20})
            if res.x is not None:
                _update(res.x, float(res.fun))
        except Exception:
            pass

        return x_best, f_best
