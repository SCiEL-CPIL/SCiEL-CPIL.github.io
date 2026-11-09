"""Katsuura (f23) minimizer: diverse sampling + tiered Nelder-Mead + high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Katsuura function (f23) in 5 dimensions.

    Strategy:
    1. Diverse initial sampling: origin, local Gaussians (two scales), and uniform points.
    2. Tiered local search via Nelder-Mead:
       - Tier 1: Deep search on top 4 candidates.
       - Tier 2: Medium search on next 6 candidates.
       - Tier 3: High-precision polish on the global best.
    3. Final local restarts if budget remains.
    4. Time and budget guards to ensure validity within 330s wall-clock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0  # Safety margin before 330s kill
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')

        rng = np.random.default_rng(12345)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x_clipped = np.clip(x, -5, 5)
            v = func(x_clipped)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x_clipped.copy()
            return v

        # --- 1. Initial Diverse Sampling ---
        init_points = []

        # Origin
        init_points.append(np.zeros(dim))

        # Local Gaussian sampling: sigma=0.25 (12 pts)
        for _ in range(12):
            p = rng.normal(0, 0.25, dim)
            init_points.append(np.clip(p, -5, 5))

        # Local Gaussian sampling: sigma=0.8 (8 pts)
        for _ in range(8):
            p = rng.normal(0, 0.8, dim)
            init_points.append(np.clip(p, -5, 5))

        # Uniform sampling: 20 pts across [-5, 5]^5
        for _ in range(20):
            p = rng.uniform(-5, 5, dim)
            init_points.append(p)

        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)

        sorted_indices = np.argsort(init_values)

        # --- 2. Tiered Local Search ---
        # Tier 1: Deep Nelder-Mead on top 4 candidates
        top_k_tier1 = 4
        for i in range(min(top_k_tier1, len(sorted_indices))):
            if calls >= self.budget - 100:
                break
            if time.time() - start_time > time_limit - 15.0:
                break
            seed_idx = sorted_indices[i]
            seed = init_points[seed_idx].copy()
            try:
                res = minimize(wrapped_func, seed, method='Nelder-Mead',
                               options={
                                   'maxiter': 80,
                                   'xatol': 1e-7,
                                   'fatol': 1e-9,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Tier 2: Medium Nelder-Mead on next 6 candidates (indices 4-9)
        top_k_tier2 = 6
        for i in range(top_k_tier1, min(top_k_tier1 + top_k_tier2, len(sorted_indices))):
            if calls >= self.budget - 60:
                break
            if time.time() - start_time > time_limit - 15.0:
                break
            seed_idx = sorted_indices[i]
            seed = init_points[seed_idx].copy()
            try:
                res = minimize(wrapped_func, seed, method='Nelder-Mead',
                               options={
                                   'maxiter': 30,
                                   'xatol': 1e-6,
                                   'fatol': 1e-8,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Tier 3: High-precision polish on the global best
        if calls < self.budget - 10 and best_x is not None:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                               options={
                                   'maxiter': 200,
                                   'xatol': 1e-12,
                                   'fatol': 1e-14,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # --- 3. Final local restarts if budget remains ---
        if calls < self.budget - 10 and best_x is not None:
            for _ in range(4):
                if calls >= self.budget - 10:
                    break
                if time.time() - start_time > time_limit - 10.0:
                    break
                perturbation = rng.normal(0, 0.15, dim)
                seed = np.clip(best_x + perturbation, -5, 5)
                try:
                    res = minimize(wrapped_func, seed, method='Nelder-Mead',
                                   options={
                                       'maxiter': 20,
                                       'xatol': 1e-6,
                                       'fatol': 1e-8,
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
