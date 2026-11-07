"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B Optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Explicit deadline guard
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Phase 1: Global scattering to find good basins
        # Rosenbrock has a narrow valley, so we need to get close to the valley first.
        # Use a mix of random points and points biased towards the known structure.
        # The Rosenbrock function f(x) = sum(100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2)
        # has its minimum at (1,1,...,1). But the task says minimizer is in [-4,4]^5
        # with hidden shift/rotation/offset. So we can't rely on (1,1,...,1).
        # We need to explore the space.

        # Let's use a more structured exploration:
        # 1. Random points
        # 2. Points along the "valley" direction if we could detect it, but we can't.
        # So we rely on L-BFGS-B's ability to navigate the valley once close.

        # Strategy:
        # - Spend ~30% of budget on scattering to find top basins
        # - Use L-BFGS-B on the best few starting points

        n_scatter = min(60, self.budget // 3)
        points = []
        fvals = []

        for _ in range(n_scatter):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            points.append(x)
            fvals.append(fx)

        if not points:
            x = np.zeros(self.dim)
            return x, func(x)

        # Sort by fitness
        indices = np.argsort(fvals)
        points = [points[i] for i in indices]
        fvals = [fvals[i] for i in indices]

        best_x = points[0].copy()
        best_f = fvals[0]

        # Phase 2: Local refinement with L-BFGS-B
        # L-BFGS-B is much more efficient than Nelder-Mead for smooth functions like Rosenbrock
        # once in the basin of attraction.

        remaining_budget = self.budget - len(points)

        # How many seeds to refine?
        # L-BFGS-B converges faster, so we can afford to try more seeds.
        # But each L-BFGS-B run uses multiple function calls internally.
        # Let's try the top 5-8 seeds.

        n_seeds_to_refine = min(8, len(points))

        # Define a wrapper to count calls and enforce limits
        call_count = 0
        max_total_calls = self.budget

        class CallCounter:
            def __init__(self):
                self.count = 0
            def __call__(self, x):
                nonlocal call_count
                if call_count >= max_total_calls:
                    return 1e10
                call_count += 1
                return func(x)

        counter = CallCounter()

        for i in range(n_seeds_to_refine):
            if time.time() > deadline:
                break

            start_x = points[i].copy()

            # L-BFGS-B with bounds
            try:
                res = minimize(
                    counter,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 50,
                        'maxfun': max(10, (max_total_calls - call_count) // max(1, n_seeds_to_refine - i))
                    }
                )

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

            except Exception:
                pass

        # If we have budget left, do a final polish with the best point
        if time.time() < deadline and call_count < max_total_calls:
            try:
                res = minimize(
                    counter,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50,
                        'maxfun': max(10, max_total_calls - call_count)
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
