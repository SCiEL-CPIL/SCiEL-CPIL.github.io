"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial exploration: a few random points to get a rough sense of the landscape
        x = np.zeros(dim)
        fx = func(x)

        # Try a few random starting points to pick the best initial
        for _ in range(5):
            x_try = rng.uniform(-5, 5, dim)
            fx_try = func(x_try)
            if fx_try < fx:
                x, fx = x_try, fx_try

        calls_made = 6  # 1 + 5

        # Adaptive (1+1)-ES with restarts
        best_x = x.copy()
        best_fx = fx

        # Parameters
        sigma = 1.5
        min_sigma = 1e-6
        max_sigma = 10.0
        patience = 30  # restart if no improvement in this many steps
        no_improve = 0

        while calls_made < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, max_sigma)
                no_improve = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma = max(sigma * 0.9, min_sigma)
                no_improve += 1

            # Restart mechanism: if stuck, jump to a new random point
            if no_improve >= patience:
                # Try a few random restarts
                for _ in range(3):
                    x_try = rng.uniform(-5, 5, dim)
                    fx_try = func(x_try)
                    calls_made += 1
                    if fx_try < fx:
                        x, fx = x_try, fx_try
                        if fx < best_fx:
                            best_x, best_fx = x.copy(), fx
                        break
                sigma = 1.5
                no_improve = 0

        # Final polish: if we have time, do a fine-tuning pass around the best point
        if time.time() < deadline - 5.0 and calls_made < budget:
            x_polish = best_x.copy()
            fx_polish = best_fx
            sigma_polish = 0.1
            for _ in range(min(50, budget - calls_made)):
                if time.time() > deadline - 2.0:
                    break
                y = np.clip(x_polish + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx_polish:
                    x_polish, fx_polish = y, fy
                    sigma_polish = min(sigma_polish * 1.1, 0.5)
                else:
                    sigma_polish = max(sigma_polish * 0.9, 1e-6)
                if fx_polish < best_fx:
                    best_x, best_fx = x_polish.copy(), fx_polish

        return best_x, best_fx
# EVOLVE-BLOCK-END
