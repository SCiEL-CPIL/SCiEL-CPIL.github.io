"""Optimized BBOB f21 (Gallagher 101 peaks) optimizer.

Strategy: cheap global quasi-random seeding to catch multiple basins, then
adaptive (1+1)-ES local refinement with a success-ratio step-size controller.
Hard budget + wall-clock guards keep the run valid and fast.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _halton(i, base):
    """Van der Corput / Halton digit-reversal in [0,1)."""
    f = 1.0
    r = 0.0
    i = int(i)
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class Optimizer:
    """Global seeding + adaptive (1+1)-ES with budget & time guards."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0
        span = hi - lo
        start = time.time()
        deadline = start + 280.0  # safety margin well under the 330 s kill

        # Deterministic quasi-random global seeds (cheap, no RNG).
        n_seed = min(40, budget)
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        seeds = []
        for i in range(1, n_seed + 1):
            seeds.append(np.array(
                [lo + span * _halton(i, bases[d]) for d in range(dim)]
            ))

        used = 0
        best = None
        best_f = math.inf

        def consider(x):
            nonlocal best, best_f, used
            fx = float(func(x))
            used += 1
            if fx < best_f:
                best_f = fx
                best = x.copy()
            return fx

        # Phase 1: global seeding.
        for s in seeds:
            if used >= budget or time.time() > deadline:
                break
            consider(s)

        # Phase 2: adaptive (1+1)-ES refinement from every seed.
        # Success-ratio controller: sigma *= (1 + 0.8*(r - 1/5))^(1/5).
        for s in seeds:
            if used >= budget or time.time() > deadline:
                break
            x = s.copy()
            fx = best_f if (best is not None and np.allclose(x, best)) else None
            if fx is None:
                fx = float(func(x))
                used += 1
                if fx < best_f:
                    best_f = fx
                    best = x.copy()
            sigma = 1.0
            n = 0
            while used < budget and time.time() < deadline:
                y = x + sigma * np.random.standard_normal(dim)
                np.clip(y, lo, hi, out=y)
                fy = float(func(y))
                used += 1
                n += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best = x.copy()
                if n % 5 == 0:
                    r = n_success / 5.0 if False else None  # placeholder
                    # recompute success ratio over the last 5 steps
                    r = self._last_success_ratio(n_success_counter, 5)
                    sigma *= (1.0 + 0.8 * (r - 0.2)) ** 0.2
                    sigma = min(max(sigma, 1e-6), 10.0)
        return best, best_f
# EVOLVE-BLOCK-END
