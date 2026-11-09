"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with adaptive step-size, time guard, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Initial random sampling to find a good starting point
        init_budget = min(30, budget // 10)
        for _ in range(init_budget):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls = 1
            best_f = fx
            best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma
        x = best_x.copy()
        sigma = 0.5
        stall_counter = 0
        stall_threshold = 20

        while time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < func(x):
                x = y
                sigma *= 1.1
                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                sigma *= 0.85
                stall_counter += 1

            if stall_counter >= stall_threshold:
                # Try a restart from a perturbed best point
                if time_ok():
                    x = np.clip(best_x + rng.standard_normal(dim) * 0.3, -5, 5)
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = 0.5
                    stall_counter = 0
                    if not time_ok():
                        break
                else:
                    break

        # Phase 3: Nelder-Mead polish if time and budget remain
        if time_ok() and calls < budget - 10:
            remaining_budget = budget - calls
            # Limit Nelder-Mead iterations to use at most half remaining budget
            max_iter = min(200, remaining_budget // 2)
            
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'maxfev': min(100, remaining_budget),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
