"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol exploration + (1+1)-ES refinement for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        rng = np.random.default_rng(42)
        call_count = [0]

        def safe_call(x):
            if call_count[0] >= budget:
                return np.inf
            call_count[0] += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # Phase 1: Sobol sequence exploration for deterministic space-filling coverage
        # Use first 100 evaluations for global exploration
        sobol_n = 100
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_samples = sampler.random(sobol_n) * 10 - 5  # Map [0,1]^d to [-5,5]^d

        for i in range(sobol_n):
            if time.time() > deadline or call_count[0] >= budget:
                break

            x = sobol_samples[i]
            f = safe_call(x)

            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: (1+1)-ES refinement from the best Sobol solution
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - call_count[0]

            for _ in range(remaining):
                if time.time() > deadline or call_count[0] >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.92

                if sigma < 1e-10:
                    break

        # Phase 3: Final L-BFGS-B polish if we have budget left
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            remaining = budget - call_count[0]
            if remaining > 10:
                try:
                    res = minimize(
                        safe_call,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': remaining, 'maxls': 10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
