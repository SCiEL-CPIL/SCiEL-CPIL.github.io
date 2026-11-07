"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple robust optimizer: random search with local refinement for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def check_time():
            return time.time() < deadline and calls < budget

        # Phase 1: Broad random search (60% of budget)
        n_broad = int(budget * 0.6)
        for _ in range(n_broad):
            if not check_time():
                break
            x = rng.uniform(-5, 5, size=dim)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        # Phase 2: Local refinement around best point (40% of budget)
        n_local = budget - calls
        sigma = 1.0
        for i in range(n_local):
            if not check_time():
                break
            # Adaptive step size: decrease over time
            sigma = 1.0 * (0.95 ** (i // 10))
            x_new = best_x + sigma * rng.standard_normal(dim)
            x_new = np.clip(x_new, -5, 5)
            fx = func(x_new)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_new.copy()
            else:
                # Occasionally try larger jumps to escape local minima
                if i % 20 == 19 and sigma > 0.1:
                    x_jump = best_x + 2.0 * rng.standard_normal(dim)
                    x_jump = np.clip(x_jump, -5, 5)
                    fx_jump = func(x_jump)
                    calls += 1
                    if fx_jump < best_fx:
                        best_fx = fx_jump
                        best_x = x_jump.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
