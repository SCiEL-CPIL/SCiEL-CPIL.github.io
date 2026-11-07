"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Two-stage hybrid: CMA-ES for global search, then L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = np.zeros(d)
        best_fx = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return best_fx
            x = np.clip(x, -5, 5)
            calls += 1
            fx = float(func(x))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # --- Stage 1: CMA-ES style global search with limited budget ---
        # Use a simple population-based search to identify the basin
        cma_budget = 120  # ~120 calls for global exploration

        # Initialize population around center with moderate spread
        rng = np.random.RandomState(42)
        pop_size = 10
        sigma = 3.0  # Initial step size
        center = np.zeros(d)

        # Generate initial population
        population = []
        for i in range(pop_size):
            x = center + sigma * rng.randn(d)
            x = np.clip(x, -5, 5)
            population.append(x)

        cma_calls = 0
        max_cma_iter = 12  # 12 iterations * 10 pop = 120 calls

        for it in range(max_cma_iter):
            if not time_ok() or calls >= cma_budget:
                break

            # Evaluate population
            fvals = []
            for x in population:
                fx = safe_func(x)
                fvals.append(fx)

            # Sort by fitness
            indices = np.argsort(fvals)
            population = [population[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            # Select top half for recombination
            k = pop_size // 2
            weights = np.exp(-np.arange(k) / (k / 2.0))
            weights /= weights.sum()

            # Weighted mean (recombination)
            new_center = np.zeros(d)
            for w, x in zip(weights, population[:k]):
                new_center += w * x

            # Update sigma based on spread (simple adaptation)
            spread = np.mean([np.linalg.norm(x - new_center) for x in population[:k]])
            sigma = max(0.1, min(5.0, sigma * (1.0 if spread < 0.5 else 0.8)))

            # Generate new population around new center
            new_population = []
            for i in range(pop_size):
                x = new_center + sigma * rng.randn(d)
                x = np.clip(x, -5, 5)
                new_population.append(x)
            population = new_population

            center = new_center

        # --- Stage 2: L-BFGS-B polish from multiple starting points ---
        # Use the best points from CMA-ES plus some deterministic starts
        lbgfs_budget = budget - calls - 5  # Reserve a few calls

        # Collect starting points: best from CMA-ES, center, and a few random
        start_points = []

        # Best point found so far
        start_points.append(best_x.copy())

        # Center
        start_points.append(np.zeros(d))

        # A few points from the final CMA-ES population
        for x in population[:3]:
            if x is not best_x and not np.allclose(x, best_x):
                start_points.append(x.copy())

        # Add a couple of random points for diversity
        for _ in range(2):
            p = rng.uniform(-4, 4, size=d)
            start_points.append(p)

        # Remove duplicates
        unique_starts = []
        for p in start_points:
            is_dup = False
            for q in unique_starts:
                if np.allclose(p, q, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(p)

        bounds = [(-5, 5)] * d

        # Run L-BFGS-B from each starting point, prioritizing best first
        for x0 in unique_starts:
            if not time_ok() or calls >= budget - 3:
                break

            def local_func(v):
                nonlocal calls
                if calls >= budget:
                    return best_fx
                return safe_func(v)

            try:
                res = minimize(
                    local_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxls': 50,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'eps': 1e-9
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # --- Stage 3: Final high-precision polish if budget remains ---
        if time_ok() and calls < budget - 5:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 200,
                        'maxls': 100,
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                        'eps': 1e-10
                    }
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
