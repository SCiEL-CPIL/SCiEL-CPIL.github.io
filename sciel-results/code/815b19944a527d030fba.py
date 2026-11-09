"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, then Nelder-Mead polish within budget."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        d = self.dim
        budget = self.budget

        # Start at the center of the box
        x = np.zeros(d, dtype=np.float64)
        fx = func(x)
        calls = 1

        # Initial step size: moderate for early convergence
        sigma = 1.0

        # 1/5 success rule adaptation factors
        c_inc = 1.1
        c_dec = 1.0 / 1.1

        sigma_min = 1e-10
        sigma_max = 20.0

        # Phase 1: (1+1)-ES with per-step 1/5 adaptation
        for i in range(budget - 1):
            if time.time() > deadline:
                break

            direction = rng.standard_normal(d)
            y = np.clip(x + sigma * direction, -5.0, 5.0)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                sigma *= c_inc
            else:
                sigma *= c_dec

            # Clamp sigma
            if sigma < sigma_min:
                sigma = sigma_min
            elif sigma > sigma_max:
                sigma = sigma_max

        # Phase 2: Nelder-Mead polish with remaining budget
        remaining = budget - calls
        if remaining >= 20 and time.time() < deadline - 10.0:
            try:
                from scipy.optimize import minimize

                # Track calls during NM
                nm_calls = [0]

                def wrapped_func(v):
                    nm_calls[0] += 1
                    if nm_calls[0] > remaining:
                        return fx  # Stop by returning current best
                    return func(v)

                # Limit iterations to stay within budget
                # Nelder-Mead in d dimensions uses d+1 points per iteration
                max_iter = max(1, remaining // (d + 2))
                max_iter = min(max_iter, 100)  # Cap to avoid excessive time

                res = minimize(
                    wrapped_func, x.copy(), method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )

                if nm_calls[0] <= remaining and res.fun < fx:
                    x = np.clip(res.x, -5.0, 5.0)
                    fx = res.fun
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
