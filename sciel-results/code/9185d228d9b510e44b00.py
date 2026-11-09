"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, leave 20s margin
        rng = np.random.default_rng(0)

        best_x = None
        best_fx = np.inf
        calls_made = 0
        max_calls = self.budget

        def elapsed_ok():
            return (time.time() - start_time) < deadline and calls_made < max_calls

        # Phase 1: Broad exploration with multiple random starts
        n_starts = 10
        for s in range(n_starts):
            if not elapsed_ok():
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES from best point with adaptive sigma
        if best_x is not None and elapsed_ok():
            x = best_x.copy()
            fx = best_fx
            sigma = 1.5
            no_improve = 0
            max_no_improve = 30

            while elapsed_ok():
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.3
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                if no_improve > max_no_improve:
                    # Try a new random start
                    if elapsed_ok():
                        x = rng.uniform(-5, 5, self.dim)
                        fx = func(x)
                        calls_made += 1
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        sigma = 2.0
                        no_improve = 0
                    else:
                        break

        # Phase 3: If we have time, do a final local refinement
        if best_x is not None and elapsed_ok() and calls_made < max_calls:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.3
            for _ in range(min(50, max_calls - calls_made)):
                if not elapsed_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
