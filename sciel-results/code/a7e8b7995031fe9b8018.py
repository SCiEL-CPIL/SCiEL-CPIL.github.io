"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive step size, followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        d = self.dim
        total_budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        calls = 1

        sigma = 1.5
        success_count = 0
        total_trials = 0
        target_success = 0.2

        # (1+1)-ES phase: use ~70% of budget
        es_budget = int(total_budget * 0.7)

        for _ in range(es_budget):
            if not time_ok() or calls >= total_budget - 20:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                # Increase sigma on success
                sigma *= 1.1
            else:
                sigma *= 0.9

            total_trials += 1
            # Adapt sigma based on success rate over a window
            if total_trials % 20 == 0:
                recent_success = success_count / total_trials
                if recent_success > target_success * 1.5:
                    sigma *= 1.2
                elif recent_success < target_success * 0.5:
                    sigma *= 0.8
                # Reset counters
                success_count = 0
                total_trials = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 5.0)

        # Nelder-Mead polish phase: use remaining budget
        if time_ok() and calls < total_budget - 10:
            nm_budget = total_budget - calls

            def nm_func(x_vec):
                nonlocal calls
                if not time_ok() or calls >= total_budget:
                    return fx  # Return best known to stop
                x_vec = np.clip(x_vec, -5, 5)
                val = func(x_vec)
                calls += 1
                return val

            # Set up Nelder-Mead with a reasonable initial simplex
            # The simplex size should be small since we're polishing
            nm_sigma = max(sigma, 0.01)
            initial_simplex = [x.copy()]
            for i in range(d):
                perturbation = x.copy()
                perturbation[i] += nm_sigma
                initial_simplex.append(perturbation)

            try:
                result = minimize(
                    nm_func,
                    x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget,
                        'adaptive': True,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'initial_simplex': np.array(initial_simplex)
                    }
                )
                if result.fun < fx:
                    x = np.clip(result.x, -5, 5)
                    fx = result.fun
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
