"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES-like optimizer with adaptive sigma, stall detection, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Multi-restart strategy: split budget among restarts
        # Use more restarts for multi-modal functions
        num_restarts = min(8, max(3, budget // 60))
        budget_per_restart = budget // num_restarts

        for restart_idx in range(num_restarts):
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize
            if restart_idx == 0:
                # First restart: try center and a few random points
                x = np.zeros(dim)
                fx = func(x)
                total_calls += 1
                if fx < best_fx:
                    best_fx, best_x = fx, x.copy()

                # Try a few random starting points
                for _ in range(min(3, budget - total_calls)):
                    if total_calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    x_try = rng.uniform(-4, 4, dim)
                    fx_try = func(x_try)
                    total_calls += 1
                    if fx_try < best_fx:
                        best_fx, best_x = fx_try, x_try.copy()

                x = best_x.copy()
                fx = best_fx
            else:
                # Subsequent restarts: perturb from best
                if best_x is not None:
                    x = best_x + rng.normal(0, 1.0, dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-4, 4, dim)
                fx = func(x)
                total_calls += 1
                if fx < best_fx:
                    best_fx, best_x = fx, x.copy()

            # Run (1+1)-ES with adaptive step size
            sigma = 1.0
            stall_count = 0
            stall_limit = max(15, dim * 3)
            remaining = budget - total_calls
            if remaining <= 0:
                break

            while total_calls < budget:
                if time.time() > deadline:
                    break

                # Adaptive sigma: shrink if stalled, grow if improving
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                    sigma *= 1.15  # grow step on success
                    stall_count = 0
                else:
                    sigma *= 0.85  # shrink step on failure
                    stall_count += 1

                # Stall detection: restart locally
                if stall_count >= stall_limit:
                    # Try a local restart around current best
                    if best_x is not None:
                        x = best_x + rng.normal(0, 0.5, dim)
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        total_calls += 1
                        if fx < best_fx:
                            best_fx, best_x = fx, x.copy()
                    else:
                        x = rng.uniform(-4, 4, dim)
                        fx = func(x)
                        total_calls += 1
                        if fx < best_fx:
                            best_fx, best_x = fx, x.copy()
                    sigma = 0.5
                    stall_count = 0
                    stall_limit = max(10, dim * 2)

                # If sigma becomes too small, do a final polish with smaller steps
                if sigma < 0.01:
                    # Polish phase
                    for _ in range(min(10, budget - total_calls)):
                        if total_calls >= budget or time.time() > deadline:
                            break
                        y = x + sigma * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        total_calls += 1
                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_fx:
                                best_fx, best_x = fx, x.copy()
                            sigma *= 1.1
                        else:
                            sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
