"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES-like optimizer with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Phase 1: Random restarts to find a good basin
        # Rastrigin has many local minima; random restarts help
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local optimization from best point using adaptive step
        if best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            no_improve = 0
            max_no_improve = 20

            while budget > 0 and time.time() < deadline:
                # Generate candidate
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                # If stuck, try a larger jump or restart locally
                if no_improve >= max_no_improve:
                    # Try a larger random jump
                    for _ in range(5):
                        if time.time() > deadline or budget <= 0:
                            break
                        y = np.clip(x + sigma * 3 * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        budget -= 1
                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            sigma *= 1.5
                            no_improve = 0
                            break
                    else:
                        # Still stuck, do a local restart
                        x = rng.uniform(-4, 4, dim)
                        fx = func(x)
                        budget -= 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma = 1.0
                        no_improve = 0

        # Phase 3: Fine-tuning with very small steps if budget remains
        if best_x is not None and budget > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(50, budget)):
                if time.time() > deadline:
                    break
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                budget -= 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
