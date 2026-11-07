# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved portfolio search: diverse random restarts with adaptive (1+1)-ES,
       coordinate-wise refinement, and final high-precision polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration with diverse starting points
        x = rng.uniform(-5, 5, dim)
        fx = evaluate(x)

        sigma_explore = 2.5
        n_explore = min(60, budget // 4)
        for i in range(n_explore):
            if calls_made >= budget or remaining_time() < 5.0:
                break
            # Mix: 30% pure random, 70% perturbation of best
            if rng.random() < 0.3:
                y = clip(rng.uniform(-5, 5, dim))
            else:
                y = clip(best_x + sigma_explore * rng.standard_normal(dim))

            fy = evaluate(y)
            if fy < best_f - 0.5:
                sigma_explore = min(3.5, sigma_explore * 1.15)
            elif i > 0 and i % 8 == 0:
                sigma_explore = max(0.8, sigma_explore * 0.92)

        # Phase 2: Multi-start (1+1)-ES with adaptive step size
        max_restarts = 10
        remaining_calls = budget - calls_made
        calls_per_restart = max(25, remaining_calls // max_restarts)

        for restart in range(max_restarts):
            if calls_made >= budget or remaining_time() < 5.0:
                break

            # Choose starting point: diversify based on restart number
            if restart == 0:
                x = best_x.copy()
            elif restart == 1:
                x = clip(best_x + 2.0 * rng.standard_normal(dim))
            elif restart == 2:
                x = clip(best_x + 3.0 * rng.standard_normal(dim))
            elif rng.random() < 0.25:
                x = clip(rng.uniform(-5, 5, dim))
            else:
                scale = 1.0 + 0.5 * restart
                x = clip(best_x + scale * rng.standard_normal(dim))

            fx = evaluate(x)
            sigma = 0.5
            success_count = 0
            total_count = 0
            stagnant = 0

            # (1+1)-ES with adaptive step size
            for i in range(calls_per_restart):
                if calls_made >= budget or remaining_time() < 3.0:
                    break

                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                total_count += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    stagnant = 0
                    sigma *= 1.08
                    if sigma > 2.0:
                        sigma = 2.0
                else:
                    sigma *= 0.93
                    stagnant += 1
                    if sigma < 0.003:
                        sigma = 0.003

                # Adaptive step size based on success rate
                if total_count % 12 == 0 and total_count >= 12:
                    success_rate = success_count / total_count
                    if success_rate > 0.4:
                        sigma *= 1.3
                    elif success_rate < 0.15:
                        sigma *= 0.75
                    success_count = 0
                    total_count = 0

                # Early termination if stagnant
                if stagnant > 30:
                    break

            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: Coordinate-wise refinement around best point
        if calls_made < budget and remaining_time() > 5.0:
            x = best_x.copy()
            fx = best_f
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
            for step in step_sizes:
                if calls_made >= budget or remaining_time() < 3.0:
                    break
                for coord in range(dim):
                    if calls_made >= budget or remaining_time() < 3.0:
                        break
                    for direction in [-1, 1]:
                        if calls_made >= budget or remaining_time() < 3.0:
                            break
                        y = x.copy()
                        y[coord] += direction * step
                        y = clip(y)
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy
                        else:
                            break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 4: Final high-precision polish
        if calls_made < budget and remaining_time() > 3.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.02
            n_fine = min(80, budget - calls_made)
            for i in range(n_fine):
                if calls_made >= budget or remaining_time() < 2.0:
                    break
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.03
                else:
                    sigma *= 0.90
                    if sigma < 0.001:
                        sigma = 0.001

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
