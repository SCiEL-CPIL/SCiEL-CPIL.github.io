"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with time guard and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Random restarts to find a good starting point
        # Use a few random evaluations to get a sense of the landscape
        best_x = None
        best_f = np.inf
        num_restarts = 5
        calls_made = 0

        for r in range(num_restarts):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: (1+1)-ES with adaptive step size
        sigma = 2.0
        x = best_x.copy()
        fx = best_f

        # Track success rate for step size adaptation
        successes = 0
        window_size = 15

        while calls_made < budget and time_ok():
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Periodically adapt sigma based on success rate
            if calls_made % window_size == 0:
                success_rate = successes / window_size
                if success_rate > 0.5:
                    sigma *= 1.3
                elif success_rate < 0.2:
                    sigma *= 0.7
                successes = 0

            # Ensure sigma stays reasonable
            sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 3: Nelder-Mead polish with remaining budget
        if calls_made < budget and time_ok():
            remaining = budget - calls_made
            if remaining > 10:
                try:
                    # Use a subset of remaining budget for NM
                    nm_budget = min(remaining, 100)
                    result = minimize(
                        func,
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    calls_made += result.get('nfev', nm_budget)
                    if result.fun < fx:
                        x, fx = result.x, result.fun
                except Exception:
                    pass

        # Phase 4: If time remains, do another round of local search
        if calls_made < budget and time_ok():
            remaining = budget - calls_made
            if remaining > 20:
                # Fine-tune with small steps
                sigma = 0.1
                for _ in range(min(remaining, 50)):
                    if not time_ok() or calls_made >= budget:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.9

        return x, fx
# EVOLVE-BLOCK-END
