"""Fresh design: multi-start + adaptive (1+1)-ES with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(12345)
        deadline = time.time() + 280.0  # safety margin below the 330s kill

        # ---- Phase 1: cheap multi-start to find a good basin early ----
        # Deterministic diverse seeds: center, box-center, and a few randoms.
        n_starts = 8
        seeds = [np.zeros(dim), np.full(dim, 0.0)]
        for _ in range(n_starts - 2):
            seeds.append(rng.uniform(-4.5, 4.5, dim))
        # De-duplicate (avoid duplicate calls)
        seen = []
        uniq = []
        for s in seeds:
            key = tuple(np.round(s, 6))
            if key not in seen:
                seen.append(key)
                uniq.append(s)

        best_x = None
        best_f = np.inf
        for s in uniq:
            if best_x is None or time.time() > deadline:
                if best_x is None:
                    # must make at least one call
                    pass
                if time.time() > deadline and best_x is not None:
                    break
            fx = func(np.clip(s, -5.0, 5.0))
            if fx < best_f:
                best_f = fx
                best_x = s.copy()
            if best_x is not None and time.time() > deadline:
                break

        # ---- Phase 2: adaptive (1+1)-ES from the best start ----
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # start moderate; adapt quickly
        min_sigma, max_sigma = 1e-6, 6.0
        success = 0
        # 1/5 success rule target
        target = 1.0 / 5.0
        # adaptation: sigma *= exp(c * (success_rate - target)) each step
        c_sigma = 0.2

        n_calls = 1 + len(uniq) if best_x is not None else 0
        # We already made len(uniq) calls.
        n_calls = len(uniq)
        for _ in range(self.budget - n_calls):
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                success += 1
            # sigma adaptation via 1/5 success rule (per step, simple form)
            sigma *= np.exp(c_sigma * ((1.0 if fy < fx else 0.0) - target))
            sigma = min(max(sigma, min_sigma), max_sigma)

        return x, fx
# EVOLVE-BLOCK-END
