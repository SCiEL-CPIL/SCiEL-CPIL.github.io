"""Combined optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with per-step adaptive step-size.

    Phase 1: cheap multi-start exploration with short local descents.
    Phase 2: committed (1+1)-ES from the best point with fine-grained
    step-size adaptation (1.01/0.99 per step) for smooth convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_run(self, func, x, fx, sigma, n_calls, rng, deadline):
        """Run (1+1)-ES with per-step success rule for n_calls evaluations.

        Returns (x, fx, sigma, total_calls_used).
        """
        dim = self.dim
        used = 0
        for _ in range(n_calls):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            used += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.01  # small increase on success
            else:
                sigma *= 0.99  # small decrease on failure
            # Keep sigma in a reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)
        return x, fx, sigma, used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        # ---- Phase 1: multi-start exploration ----
        n_starts = 10
        phase1 = min(100, max(1, budget // 5))
        per_start = max(3, phase1 // n_starts)

        best_x = None
        best_f = np.inf
        calls = 0

        for _ in range(n_starts):
            if calls >= phase1 or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            # short local descent from this start
            x, fx, sigma, used = self._es_run(
                func, x, fx, 1.5, per_start, rng, deadline
            )
            calls += used
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # ---- Phase 2: committed (1+1)-ES from best point ----
        remaining = max(0, budget - calls)
        if remaining > 0:
            x, fx, sigma, used = self._es_run(
                func, best_x, best_f, 2.0, remaining, rng, deadline
            )
            calls += used
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
