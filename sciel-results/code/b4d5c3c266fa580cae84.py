"""Multi-start Nelder-Mead with CMA-ES for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local optimization for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        calls = 0

        def try_eval(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            return f

        rng = np.random.default_rng(2024)

        # Phase 1: Broad random sampling to find promising basins
        n_random = min(60, budget)
        best_f = np.inf
        best_x = None
        candidates = []

        for i in range(n_random):
            if time.time() > deadline or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = try_eval(x)
            if f is None:
                break
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start Nelder-Mead from top-k diverse points
        # Select top points that are sufficiently diverse
        top_k = min(15, len(candidates))
        diverse_starts = []
        for f, x in candidates:
            if len(diverse_starts) >= top_k:
                break
            # Check diversity: minimum distance from already selected starts
            diverse = True
            for _, dx in diverse_starts:
                if np.linalg.norm(x - dx) < 1.5:
                    diverse = False
                    break
            if diverse:
                diverse_starts.append((f, x))

        # If not enough diverse points, add more from the list
        if len(diverse_starts) < min(8, top_k):
            for f, x in candidates:
                if len(diverse_starts) >= min(8, top_k):
                    break
                if (f, x) not in diverse_starts:
                    diverse_starts.append((f, x))

        # Nelder-Mead settings
        nm_options = {
            'maxiter': 200,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True,
        }

        # Run Nelder-Mead from each diverse start
        for f_init, x_init in diverse_starts:
            if time.time() > deadline or calls >= budget:
                break

            # Wrap func with budget/time guard
            def nm_func(x):
                if calls >= budget or time.time() > deadline:
                    return 1e10
                x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                f = float(func(x))
                calls += 1
                return f

            try:
                res = minimize(nm_func, x_init, method='Nelder-Mead', options=nm_options)
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

            # If we still have budget, do a few more random restarts
            if calls < budget and time.time() < deadline:
                n_extra = min(5, budget - calls)
                for _ in range(n_extra):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_rand = rng.uniform(-5, 5, dim)
                    f_rand = try_eval(x_rand)
                    if f_rand is None:
                        break
                    if f_rand < best_f:
                        best_f = f_rand
                        best_x = x_rand.copy()

        # Phase 3: If budget remains, do a focused CMA-ES around the best point
        if calls < budget and time.time() < deadline and best_x is not None:
            try:
                from cma import CMA  # May not be available; fallback below
            except ImportError:
                # Simple local refinement with small perturbations
                sigma_refine = 0.1
                for _ in range(min(50, budget - calls)):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_try = best_x + sigma_refine * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = try_eval(x_try)
                    if f_try is None:
                        break
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                    # Adaptive shrink
                    if f_try < best_f:
                        sigma_refine *= 0.9
                    else:
                        sigma_refine *= 1.1
                        sigma_refine = min(sigma_refine, 1.0)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
