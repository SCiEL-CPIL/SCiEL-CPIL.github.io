"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: (1+1)-ES for global exploration, then Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nm_step(self, x, sigma):
        """Try Nelder-Mead from current point with small initial simplex."""
        if self.calls_made >= self.budget - 1:
            return x, self.fx
        try:
            res = minimize(
                self.func,
                x,
                method='Nelder-Mead',
                options={'maxiter': 50, 'maxfev': 30,
                         'xatol': 1e-6, 'fatol': 1e-8,
                         'adaptive': True}
            )
            if res.fun < self.fx:
                self.fx = res.fun
                x = res.x
        except Exception:
            pass
        return x, self.fx

    def __call__(self, func):
        self.func = func
        self.budget = self.budget
        self.dim = self.dim
        self.calls_made = 0
        self.deadline = time.time() + 290.0  # 300s limit, 10s safety margin

        def _call(x):
            self.calls_made += 1
            return func(x)

        # Start at origin
        x = np.zeros(self.dim)
        fx = _call(x)
        self.fx = fx

        # Phase 1: (1+1)-ES with conservative adaptation
        sigma = 2.0
        rng = np.random.default_rng(42)

        # Track best
        best_x = x.copy()
        best_fx = fx

        # Run ES for a portion of the budget, then refine
        es_budget = int(self.budget * 0.6)  # 300 calls for ES

        while self.calls_made < es_budget:
            if time.time() > self.deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = _call(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.92

            # Periodically try NM refinement
            if self.calls_made % 50 == 0 and self.calls_made < es_budget - 40:
                x, fx = self._nm_step(x, sigma)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 2: Nelder-Mead refinement from best point
        if self.calls_made < self.budget - 10:
            x, fx = self._nm_step(best_x, 1.0)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 3: Final polish if budget remains
        if self.calls_made < self.budget - 5:
            x, fx = self._nm_step(best_x, 0.5)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # One more ES pass with very small sigma to fine-tune
        if self.calls_made < self.budget - 2:
            sigma = 0.05
            x = best_x.copy()
            fx = best_fx
            while self.calls_made < self.budget - 1:
                if time.time() > self.deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = _call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
