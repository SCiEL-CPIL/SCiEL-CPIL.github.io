"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear-function-aware optimizer: estimate gradient, jump to optimal vertex, refine."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Track best
        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Estimate the linear function
        # Use a larger step for better gradient estimation
        h = 2.0
        center = np.zeros(n)
        f0 = safe_call(center)
        if f0 is None:
            return best_x, best_f

        # Estimate gradient using central differences
        grad = np.zeros(n)
        for i in range(n):
            if time.time() > deadline or call_count >= budget:
                break
            x_plus = center.copy()
            x_plus[i] += h
            x_minus = center.copy()
            x_minus[i] -= h
            fp = safe_call(x_plus)
            if fp is None:
                break
            fm = safe_call(x_minus)
            if fm is None:
                break
            grad[i] = (fp - fm) / (2.0 * h)

        if time.time() > deadline or call_count >= budget:
            return best_x, best_f

        # Phase 2: Compute the optimal vertex
        vertex = np.where(grad > 0, -5.0, 5.0)
        vertex[np.abs(grad) < 1e-12] = 0.0
        fv = safe_call(vertex)

        if time.time() > deadline or call_count >= budget:
            return best_x, best_f

        # Phase 3: Local refinement around the best point
        rng = np.random.default_rng(123)
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        min_sigma = 0.001

        while call_count < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
            fy = safe_call(y)
            if fy is None:
                break
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.1, 5.0)
            else:
                sigma = max(sigma * 0.9, min_sigma)
                if sigma <= min_sigma:
                    # Try a few more with slightly larger steps
                    for _ in range(5):
                        if call_count >= budget or time.time() > deadline:
                            break
                        y2 = np.clip(x + 0.1 * rng.standard_normal(n), -5.0, 5.0)
                        fy2 = safe_call(y2)
                        if fy2 is None:
                            break
                        if fy2 < fx:
                            x, fx = y2, fy2
                            sigma = 0.1
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
