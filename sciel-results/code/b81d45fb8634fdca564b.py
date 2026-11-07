"""Black-box continuous optimizer for BBOB-style functions using a hybrid strategy."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: CMA-ES + L-BFGS-B + coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_f = float('inf')
        calls_made = 0

        def tracked_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                return float('inf')
            calls_made += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = float(func(x_clipped))
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Phase 1: CMA-ES for global exploration
        if calls_made < self.budget and time.time() < deadline - 30:
            try:
                from cma import CMA
            except ImportError:
                CMA = None

            if CMA is not None:
                # Start CMA-ES from origin with reasonable sigma
                x0 = np.zeros(self.dim)
                sigma0 = 2.0
                opts = {
                    'maxiter': 50,
                    'maxfevals': self.budget // 2,
                    'bounds': [(-5, 5)] * self.dim,
                    'seed': 42,
                    'verbose': -9,
                    'tolfun': 1e-10,
                    'tolfunint': 1e-10,
                    'tolfunup': 1e-10,
                }
                es = CMA(x0, sigma0, opts)
                while es.notFinished() and calls_made < self.budget - 50:
                    xs = es.ask()
                    fs = [tracked_func(x) for x in xs]
                    es.tell(xs, fs)
                    if time.time() > deadline - 30:
                        break

                if es.result.x is not None:
                    x_cma = np.clip(es.result.x, -5, 5)
                    f_cma = tracked_func(x_cma)
                    if f_cma < best_f:
                        best_f = f_cma
                        best_x = x_cma.copy()
            else:
                # Fallback: multi-restart L-BFGS-B if CMA not available
                n_restarts = 10
                for restart in range(n_restarts):
                    if calls_made >= self.budget - 50 or time.time() > deadline - 30:
                        break
                    if restart == 0:
                        x0 = np.zeros(self.dim)
                    else:
                        x0 = rng.uniform(-4, 4, self.dim)
                    try:
                        result = minimize(
                            tracked_func, x0, method='L-BFGS-B',
                            bounds=[(-5, 5)] * self.dim,
                            options={'maxiter': 100, 'maxfun': 30, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = np.clip(result.x, -5, 5).copy()
                    except Exception:
                        continue

        # Phase 2: Local refinement with L-BFGS-B from best point
        if calls_made < self.budget - 20 and time.time() < deadline - 20:
            try:
                result = minimize(
                    tracked_func, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': 40, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead for final polish
        if calls_made < self.budget - 10 and time.time() < deadline - 10:
            try:
                result = minimize(
                    tracked_func, best_x, method='Nelder-Mead',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfev': 30, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 4: Coordinate descent with adaptive step
        if calls_made < self.budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            step = 0.5
            for _ in range(50):
                if calls_made >= self.budget or time.time() > deadline - 5:
                    break
                improved = False
                for coord in range(self.dim):
                    for direction in [-1, 1]:
                        if calls_made >= self.budget:
                            break
                        x_trial = x.copy()
                        x_trial[coord] = np.clip(x_trial[coord] + direction * step, -5, 5)
                        f_trial = tracked_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            step *= 1.1
                            improved = True
                        else:
                            step *= 0.8
                if not improved:
                    step *= 0.7
                    if step < 1e-6:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
