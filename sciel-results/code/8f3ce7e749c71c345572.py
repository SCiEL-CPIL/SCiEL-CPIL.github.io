"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with multi-start and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start exploration
        # Use 3 different initializations with different step sizes
        best_x = None
        best_fx = np.inf
        calls_made = 0

        # Different starting points and step sizes for diversity
        starts = [
            (rng.uniform(-5, 5, dim), 2.0),
            (rng.uniform(-3, 3, dim), 1.5),
            (np.zeros(dim), 1.0),
        ]

        # Allocate budget: 40% for multi-start, 60% for refinement
        exploration_budget = int(budget * 0.4)
        per_start_budget = max(10, exploration_budget // len(starts))

        for x0, sigma0 in starts:
            if calls_made >= exploration_budget or time.time() > deadline:
                break

            x = x0.copy()
            sigma = sigma0
            fx = func(x)
            calls_made += 1

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Run (1+1)-ES for this start
            for _ in range(per_start_budget - 1):
                if calls_made >= exploration_budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3  # More conservative growth
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9  # Faster decay

        # Phase 2: Refinement around best point
        if best_x is not None and calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5  # Start with smaller step for refinement

            # Adaptive step size based on success rate
            success_count = 0
            total_count = 0
            window_size = 20

            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    pass

                total_count += 1

                # Adapt step size based on recent success rate
                if total_count >= window_size:
                    success_rate = success_count / total_count
                    # Target ~20% success rate for optimal progress
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    else:
                        sigma *= 0.95

                    # Reset counters
                    success_count = 0
                    total_count = 0

                # Gradually decrease sigma as we approach the end
                remaining_ratio = (budget - calls_made) / max(1, budget - calls_made)
                if remaining_ratio < 0.2:
                    sigma *= 0.98

        # Phase 3: Final local polish with very small steps
        if best_x is not None and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.05

            for _ in range(50):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
