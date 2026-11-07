"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and local refinement for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Multiple restarts with (1+1)-ES
        # Weierstrass function is highly multimodal, so restarts help escape local minima
        num_restarts = 5
        calls_per_restart = (self.budget - 50) // num_restarts  # reserve 50 for final polish

        for r in range(num_restarts):
            if check_time() or calls_made >= self.budget:
                break

            # Initialize near best found so far with some perturbation, or random for first
            if best_x is None:
                x = rng.uniform(-5.0, 5.0, self.dim)
            else:
                # Perturb around best with increasing spread
                spread = 2.0 + r * 0.5
                x = np.clip(best_x + spread * rng.standard_normal(self.dim), -5.0, 5.0)

            fx = float(func(x))
            calls_made += 1
            update_best(x, fx)

            sigma = 1.5
            success_count = 0
            total_count = 0

            for i in range(calls_per_restart):
                if check_time() or calls_made >= self.budget:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = float(func(y))
                calls_made += 1
                total_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    success_count += 1
                    update_best(x, fx)
                else:
                    sigma *= 0.9

                # Adaptive step size based on success rate (1/5 rule variant)
                if total_count >= 10:
                    success_rate = success_count / total_count
                    if success_rate > 0.4:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.85
                    # Reset counters
                    success_count = 0
                    total_count = 0

        # Phase 2: Local refinement around the best point found
        # Use a finer-grained search with smaller step sizes
        if best_x is not None and calls_made < self.budget and not check_time():
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5

            # Try coordinate-wise local search
            for coord in range(self.dim):
                if check_time() or calls_made >= self.budget:
                    break
                # Try moving in positive and negative direction
                for direction in [1, -1]:
                    if check_time() or calls_made >= self.budget:
                        break
                    step = 0.5
                    for _ in range(10):
                        if check_time() or calls_made >= self.budget:
                            break
                        y = x.copy()
                        y[coord] = np.clip(y[coord] + direction * step, -5.0, 5.0)
                        fy = float(func(y))
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            update_best(x, fx)
                            step *= 0.7
                        else:
                            break

            # Fine-tuning with small random steps
            sigma = 0.1
            for i in range(min(30, self.budget - calls_made)):
                if check_time() or calls_made >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = float(func(y))
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(self.dim)
            best_fx = float(func(x))
            best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
