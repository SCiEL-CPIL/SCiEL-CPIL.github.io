"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead + L-BFGS-B with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        # Wrap func to count calls and clip
        self._calls = 0

        def wrapped(x):
            self._calls += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x)
            return float(val)

        # Multi-start: combine random points and a few heuristic starts
        rng = np.random.default_rng(42)
        dim = self.dim

        # Generate candidate starting points
        starts = []

        # Random starts
        for _ in range(15):
            starts.append(rng.uniform(-4, 4, dim))

        # Heuristic starts for Rosenbrock-like: near (1,1,...,1) and (0,0,...,0)
        starts.append(np.ones(dim))
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, 0.5))
        starts.append(np.full(dim, -0.5))

        # Deduplicate-ish by just keeping list
        best_x = None
        best_f = np.inf

        # Evaluate all starts
        for s in starts:
            if self._calls >= self.budget:
                break
            if time.time() > deadline:
                break
            f = wrapped(s)
            if f < best_f:
                best_f = f
                best_x = s.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        # Phase 1: Nelder-Mead from best start and top few
        # Sort starts by f
        start_results = []
        for s in starts:
            if self._calls >= self.budget or time.time() > deadline:
                break
            f = wrapped(s)
            start_results.append((f, s.copy()))

        start_results.sort(key=lambda x: x[0])

        # Run Nelder-Mead on top starts
        nm_options = {
            'maxiter': 200,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True,
        }

        for f0, s0 in start_results[:8]:
            if self._calls >= self.budget or time.time() > deadline:
                break

            try:
                res = minimize(
                    wrapped, s0, method='Nelder-Mead',
                    options=nm_options
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: L-BFGS-B polish from best point and a few neighbors
        lbfgs_options = {
            'maxiter': 100,
            'ftol': 1e-12,
            'gtol': 1e-8,
        }

        polish_starts = [best_x.copy()]
        # Add small perturbations around best
        for _ in range(5):
            pert = best_x + 0.1 * rng.standard_normal(dim)
            polish_starts.append(np.clip(pert, -5, 5))

        for ps in polish_starts:
            if self._calls >= self.budget or time.time() > deadline:
                break

            try:
                res = minimize(
                    wrapped, ps, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options=lbfgs_options
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a few more Nelder-Mead from random
        # points in the basin around best
        for _ in range(5):
            if self._calls >= self.budget or time.time() > deadline:
                break

            s = best_x + 0.5 * rng.standard_normal(dim)
            s = np.clip(s, -5, 5)
            try:
                res = minimize(
                    wrapped, s, method='Nelder-Mead',
                    options=nm_options
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Final polish
        if self._calls < self.budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options=lbfgs_options
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
