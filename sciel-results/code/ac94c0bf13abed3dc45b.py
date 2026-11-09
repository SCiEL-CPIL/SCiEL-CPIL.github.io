"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _es_run(self, func, x0, sigma0, budget, rng, deadline):
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0

        # 1/5 success rule parameters
        alpha = 1.0 / (dim + 2)
        c_sigma = min(1.0 / (dim + 1), 2.0 / (dim + 2))

        for _ in range(budget):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= np.exp(c_sigma * (1 - 1))  # success
                # 1/5 rule: increase sigma on success
                sigma *= 1.5 ** (1 / (dim + 1))
            else:
                # decrease sigma on failure
                sigma *= 1.5 ** (-1 / (dim + 1))

            if sigma < 1e-10:
                break

        return x, fx

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: divide budget among several restarts
        n_restarts = min(4, max(1, budget // 100))
        budget_per_restart = budget // n_restarts
        remaining = budget - n_restarts * budget_per_restart

        best_x = None
        best_fx = np.inf

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if r == 0:
                x0 = np.array([0.0] * dim)
                sigma0 = 2.0
            else:
                if best_x is not None:
                    x0 = np.clip(best_x + 1.5 * rng.standard_normal(dim), -5, 5)
                    sigma0 = 1.0
                else:
                    x0 = rng.uniform(-5, 5, dim)
                    sigma0 = 2.0

            b = budget_per_restart + (remaining if r == n_restarts - 1 else 0)
            x, fx = self._es_run(func, x0, sigma0, b, rng, deadline)
            if fx < best_fx:
                best_fx = fx
                best_x = x

        # Final polish
        if time.time() < deadline - 5 and best_x is not None:
            x_pol, fx_pol = self._es_run(func, best_x, 0.05, min(100, budget // 5), rng, deadline)
            if fx_pol < best_fx:
                best_fx = fx_pol
                best_x = x_pol

        return best_x, best_fx
# EVOLVE-BLOCK-END
