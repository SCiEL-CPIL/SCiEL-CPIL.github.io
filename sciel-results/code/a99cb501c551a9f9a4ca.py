"""Bootstrap portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer for Katsuura:
    1) Latin Hypercube sampling for fast initial basin discovery.
    2) (1+1)-ES with adaptive sigma and periodic restarts.
    3) Deadline guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._deadline_margin = 10.0  # seconds to spare before hard limit

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube sampling for fast initial discovery
        n_lhs = min(50, budget // 10)
        best_x = None
        best_f = np.inf

        for _ in range(n_lhs):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma and restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        restart_interval = 30
        no_improve_count = 0
        calls_made = n_lhs

        while calls_made < budget:
            # Deadline check
            if time.time() - start_time > 320.0:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                best_x = x.copy()
                best_f = fx
                no_improve_count = 0
                # Increase sigma on success
                sigma *= 1.2
            else:
                no_improve_count += 1
                # Decrease sigma on failure
                sigma *= 0.85

            # Periodic restart from best point with perturbation
            if no_improve_count >= restart_interval:
                # Restart from best point with larger perturbation
                x = np.clip(best_x + 2.0 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls_made += 1
                no_improve_count = 0
                sigma = 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Adaptive restart interval
            if no_improve_count > 60:
                restart_interval = max(10, restart_interval // 2)

        return best_x, best_f
# EVOLVE-BLOCK-END
