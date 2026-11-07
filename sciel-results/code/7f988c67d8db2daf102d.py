"""Optimizer for BBOB-style Sphere f1: black-box minimizer on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize as _scipy_minimize


class Optimizer:
    """Multi-start Nelder-Mead with a hard wall-clock + call-count guard."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 250.0          # safety margin well under 330s

        calls = 0

        def wrapped(x):
            nonlocal calls
            calls += 1
            if calls > budget:            # never exceed the recorded-call budget
                return 1e12
            return float(func(x))

        # Fixed, well-spread starting points (box interior, not on the boundary)
        starts = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0][:dim], dtype=float),
            np.array([0.5, -0.5, 0.5, -0.5, 0.5][:dim], dtype=float),
        ]

        best_x = starts[0].copy()
        best_f = wrapped(best_x)
        best_x = np.clip(best_x, -5.0, 5.0)

        nm_kwargs = dict(
            method="Nelder-Mead",
            options={
                "maxiter": 800,
                "xatol": 1e-9,
                "fatol": 1e-12,
                "adaptive": True,
            },
        )

        for s in starts:
            if time.time() > deadline:
                break
            x0 = np.clip(s.copy(), -5.0, 5.0)
            try:
                res = _scipy_minimize(wrapped, x0, **nm_kwargs)
            except Exception:
                # Keep the calls already made; fall back to the best seen so far.
                break
            if res.x is not None and np.isfinite(res.fun):
                rx = np.clip(np.asarray(res.x, dtype=float), -5.0, 5.0)
                rf = float(res.fun)
                if rf < best_f:
                    best_f = rf
                    best_x = rx
            if calls >= budget - 2:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
