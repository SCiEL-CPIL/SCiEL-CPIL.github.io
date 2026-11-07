"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-stage optimizer: global random sampling, Nelder-Mead local
    refinement, and adaptive (1+1)-ES polishing, with an explicit deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        DEADLINE = t_start + 285.0  # 15s safety margin before 300s wall-clock limit

        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return 1e10
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        # Phase 1: Global random sampling to find good basins
        rng = np.random.default_rng(42)
        n_global = min(50, budget // 10)
        best_x = None
        best_f = np.inf

        for _ in range(n_global):
            if time.time() > DEADLINE:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # Phase 2: Nelder-Mead local refinement from best point
        if time.time() < DEADLINE:
            try:
                res = minimize(safe_call, best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_x = res.x
                    best_f = res.fun
            except Exception:
                pass

        # Phase 3: Adaptive (1+1)-ES polishing
        if time.time() < DEADLINE and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            success_count = 0
            window_size = 10
            window = []

            while calls_made < budget and time.time() < DEADLINE:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    window.append(1)
                else:
                    sigma *= 0.9
                    window.append(0)

                if len(window) >= window_size:
                    success_rate = sum(window) / len(window)
                    if success_rate > 0.5:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    window = []

        # Final: return best result
        return best_x, best_f
# EVOLVE-BLOCK-END
