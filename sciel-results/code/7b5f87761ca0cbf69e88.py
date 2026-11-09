"""Katsuura (f23) minimizer: optimized sampling + tiered Nelder-Mead + adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Katsuura function (f23) in 5 dimensions.

    Strategy:
    1. Targeted Initial Sampling:
       - Origin (high prior probability for Katsuura global min).
       - Dense local Gaussians (sigma=0.5) to catch local minima near the center.
       - Moderate Gaussians (sigma=1.5) to explore intermediate structures.
       - Uniform points to handle large hidden shifts.
    2. Tiered Local Search via Nelder-Mead:
       - Tier 1: Deep convergence on the top 3 candidates.
       - Tier 2: Medium convergence on the next 5 candidates.
       - Tier 3: High-precision polish on the global best.
    3. Adaptive Restarts:
       - If budget remains, perform short, focused restarts around the current best.
    4. Strict time and budget guards to ensure validity within 330s.
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

        # --- 1. Targeted Initial Sampling ---
        # Based on insights: Denser near origin + moderate distance points
        init_points = []

        # Origin
        init_points.append(np.zeros(dim))

        # Dense local Gaussian: sigma=0.5 (10 points)
        for _ in range(10):
            p = rng.normal(0, 0.5, dim)
            init_points.append(np.clip(p, -5, 5))

        # Moderate Gaussian: sigma=1.5 (10 points)
        for _ in range(10):
            p = rng.normal(0, 1.5, dim)
            init_points.append(np.clip(p, -5, 5))

        # Uniform sampling: 10 points across [-5, 5]^5
        for _ in range(10):
            p = rng.uniform(-5, 5, dim)
            init_points.append(p)

        # Evaluate initial points
        init_values = []
        for p in init_points:
            v = wrapped_func(p)
            init_values.append(v)

        sorted_indices = np.argsort(init_values)

        # --- 2. Tiered Local Search ---
        
        # Tier 1: Deep Nelder-Mead on top 3 candidates
        top_k_tier1 = 3
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
                                   'maxiter': 60,
                                   'xatol': 1e-7,
                                   'fatol': 1e-9,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Tier 2: Medium Nelder-Mead on next 5 candidates (indices 3-7)
        top_k_tier2 = 5
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
                                   'maxiter': 150,
                                   'xatol': 1e-12,
                                   'fatol': 1e-14,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # --- 3. Adaptive Restarts ---
        # If budget remains, try a few more local searches near the best point
        # with decreasing perturbation scales to exploit local structure.
        if calls < self.budget - 10 and best_x is not None:
            scales = [0.5, 0.2, 0.1]
            for scale in scales:
                if calls >= self.budget - 10:
                    break
                if time.time() - start_time > time_limit - 10.0:
                    break
                perturbation = rng.normal(0, scale, dim)
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
