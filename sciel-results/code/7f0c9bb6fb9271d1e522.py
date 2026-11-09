"""Rewritten optimizer for BBOB-style suite."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy continuous optimizer with restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best point and value
        best_x = None
        best_f = np.inf
        calls_made = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # --- Phase 1: Random sampling for initial diversity ---
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)
            if time.time() > deadline:
                break

        # --- Phase 2: Multiple (1+1)-ES restarts with adaptive sigma ---
        restarts = 0
        max_restarts = 20

        while calls_made < budget and restarts < max_restarts:
            if time.time() > deadline:
                break

            # Choose restart point: mix of best-so-far perturbation and random
            if restarts == 0:
                x = best_x.copy()
            elif rng.random() < 0.4:
                # Perturb best with larger step
                x = best_x + 3.0 * rng.standard_normal(dim)
            elif rng.random() < 0.5:
                # Perturb best with smaller step
                x = best_x + 1.0 * rng.standard_normal(dim)
            else:
                # Random restart
                x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            update_best(x, fx)

            # Adaptive (1+1)-ES with step-size adaptation
            sigma = 0.5
            no_improve = 0
            max_no_improve = 40
            success_count = 0
            total_count = 0
            c_sigma = 1.0 / (dim + 2)
            c_c = 1.0 / (dim + 6)
            c_1 = 2.0 / ((dim + 1.7) ** 2 + 4)
            c_mu = min(0.5, c_1 / 2)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            # Path accumulators for covariance matrix adaptation (simplified)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            d_c = 1.0
            d_sigma = 1.0

            while calls_made < budget:
                if time.time() > deadline:
                    break

                # Generate step
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                calls_made += 1
                total_count += 1

                if fy < fx:
                    # Success
                    success_count += 1
                    no_improve = 0
                    # Update paths
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y - x) / sigma
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y - x) / sigma
                    # Update covariance matrix (simplified)
                    C = (1 + c_1 + c_mu - c_c) * C + c_1 * np.outer(y - x, y - x) / sigma ** 2
                    # Update step size
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                    x, fx = y, fy
                    update_best(x, fx)
                else:
                    no_improve += 1
                    # Update paths even on failure
                    p_sigma = (1 - c_sigma) * p_sigma
                    p_c = (1 - c_c) * p_c
                    C = (1 + c_1 - c_c) * C
                    # Decay step size
                    sigma *= 0.9
                    if no_improve > max_no_improve:
                        break

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 10.0)

                # Check if we should restart
                if no_improve > 30 and total_count > 50:
                    break

            restarts += 1

        # --- Phase 3: Local refinement around best point ---
        if time.time() < deadline and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            no_improve = 0
            for _ in range(budget - calls_made):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma = min(sigma * 1.1, 2.0)
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1
                    if no_improve > 20:
                        break
                    if sigma < 0.005:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
