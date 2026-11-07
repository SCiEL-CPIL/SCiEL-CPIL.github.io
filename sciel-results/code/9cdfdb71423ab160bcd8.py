import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """BBOB-style optimizer using Multi-start Powell's method."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0
        rng = np.random.default_rng(42)

        def wrapper(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exhausted")
            self.calls += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # Phase 1: Coarse random search to identify promising basins
        n_init = 10
        candidates = []
        for _ in range(n_init):
            if self.calls >= budget:
                break
            x0 = rng.uniform(-5, 5, dim)
            try:
                f0 = wrapper(x0)
                candidates.append((f0, x0.copy()))
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
            except ValueError:
                break

        if not candidates:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])

        # Phase 2: Local refinement using Powell's method on the best candidates
        # Powell is effective for Rosenbrock due to its direction set alignment.
        for i, (f0, x0) in enumerate(candidates[:3]):
            if self.calls >= budget:
                break
            
            remaining = budget - self.calls
            if remaining < 20:
                continue

            # Allocate a portion of the remaining budget to each start
            # We give more budget to the better starts
            alloc = max(20, remaining // (3 - i))
            
            try:
                result = minimize(
                    wrapper,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': 100,
                        'maxfev': alloc,
                        'xtol': 1e-8,
                        'ftol': 1e-8
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except (ValueError, Exception):
                # Budget exhausted or other error, stop trying other starts
                break

        # Phase 3: Fine-tuning with local random search if budget remains
        if best_x is not None and self.calls < budget:
            sigma = 0.05
            while self.calls < budget:
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = wrapper(y)
                except ValueError:
                    break
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.5
                else:
                    sigma *= 0.5
                    if sigma < 1e-5:
                        break

        return best_x, best_f
