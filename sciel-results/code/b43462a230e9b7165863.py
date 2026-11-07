import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Optimizer combining random initialization with L-BFGS-B and a final polish pass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Track calls to ensure we don't exceed budget
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        # Bent Cigar has a narrow valley; random sampling helps locate it
        rng = np.random.default_rng()
        n_samples = 30
        for i in range(n_samples):
            if time.time() > deadline:
                break
            if call_count >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = wrapped_func(x)

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Local optimization with L-BFGS-B
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        remaining_budget = self.budget - call_count

        try:
            if remaining_budget > 10:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'maxfev': remaining_budget, 'ftol': 1e-15, 'gtol': 1e-15}
                )
        except Exception:
            pass

        # Phase 3: Final high-precision polish pass
        # Use Powell's method which is effective for unimodal functions with narrow valleys
        remaining_budget = self.budget - call_count
        if remaining_budget > 10 and time.time() < deadline:
            try:
                result2 = minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': 500, 'maxfev': remaining_budget, 'ftol': 1e-15, 'xtol': 1e-15, 'disp': False}
                )
            except Exception:
                pass

        return best_x, best_f
