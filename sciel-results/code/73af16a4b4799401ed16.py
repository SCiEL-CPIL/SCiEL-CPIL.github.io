"""Combined optimizer: Sobol global sampling + adaptive CMA-ES-like search + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-phase optimizer: Sobol sampling, adaptive local search, and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Sobol quasi-random initialization for good coverage
        n_sobol = 64
        sobol_points = self._sobol_sequence(n_sobol, dim)
        # Scale from [0,1]^dim to [-5,5]^dim
        sobol_points = sobol_points * 10.0 - 5.0

        # Evaluate all Sobol points
        best_x = None
        best_f = np.inf
        evaluated = []

        for i in range(n_sobol):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            evaluated.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Adaptive local search around best points
        # Sort evaluated points and take top candidates
        evaluated.sort(key=lambda t: t[0])
        n_local = min(10, len(evaluated))
        local_starts = [evaluated[i][1] for i in range(n_local)]

        # Adaptive step size based on function landscape
        sigma = 1.0
        min_sigma = 1e-4
        max_sigma = 2.0

        for start_idx in range(n_local):
            if time.time() > deadline:
                break

            x = local_starts[start_idx].copy()
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive (1+1)-ES with success rule
            local_budget = 30  # evaluations per local search
            for _ in range(local_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.3, max_sigma)
                else:
                    sigma = max(sigma * 0.8, min_sigma)

        # Phase 3: L-BFGS-B polish from the best point found
        if time.time() < deadline:
            try:
                result = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Final high-precision polish if time remains
        if time.time() < deadline - 5:
            try:
                result = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f

    def _sobol_sequence(self, n, dim):
        """Generate Sobol quasi-random sequence in [0,1]^dim."""
        # Use a simple Sobol-like construction using bit-reversal
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                # Use bit-reversal of i shifted by d
                val = self._bit_reverse(i + 1, 20)  # 20 bits for good resolution
                points[i, d] = val / (2**20)
        return points

    def _bit_reverse(self, x, bits):
        """Reverse the bits of x in the given number of bits."""
        result = 0
        for i in range(bits):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result
# EVOLVE-BLOCK-END
