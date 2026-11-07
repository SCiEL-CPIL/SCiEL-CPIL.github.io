"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style adaptive step size optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Initial population
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        sigma = 2.0
        # Adaptation rates
        c_sigma = (4 + np.log(d)) / (d + 4)
        c_c = (4 + np.log(d)) / (d + 4)
        c_1 = 2 / ((d + 1.3) ** 2 + d)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (d + 2) - 1))
        mu_eff = mu_eff if 'mu_eff' in dir() else None

        # Population size
        lambda_ = 4 + int(3 * np.log(d))
        mu = lambda_ // 2
        weights = np.log(np.arange(lambda_, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        # Mean and covariance
        mean = x.copy()
        C = np.eye(d)
        # Precompute C^0.5 (initially identity)
        C_inv_sqrt = np.eye(d)

        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        calls_made = 1

        for gen in range(budget - 1):
            if not time_ok():
                break

            # Sample lambda_ offspring
            offspring = []
            offspring_vals = []

            for i in range(lambda_):
                if not time_ok():
                    break

                # Sample from N(mean, sigma^2 * C)
                # Use Cholesky-like decomposition (for simplicity, use eigendecomposition periodically)
                # For now, use a simplified approach with random directions
                z = rng.standard_normal(d)
                # Approximate C^0.5 * z using the current C
                # For efficiency, we'll use a simplified update
                y = mean + sigma * (C @ z)  # This is not exactly right but a reasonable approximation
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_vals.append(fy)

            if len(offspring) < mu:
                break

            # Sort by fitness
            order = np.argsort(offspring_vals)
            selected = [offspring[i] for i in order[:mu]]
            selected_vals = [offspring_vals[i] for i in order[:mu]]

            # Update best
            if selected_vals[0] < best_fx:
                best_fx = selected_vals[0]
                best_x = selected[0].copy()

            # Compute weighted mean of selected
            new_mean = np.zeros(d)
            for i, w in enumerate(weights[:mu]):
                new_mean += w * selected[i]

            # Update sigma (CSA)
            # Simplified: use success rule
            successes = sum(1 for v in selected_vals if v < fx)
            if successes > 0:
                sigma *= (1.0 + 0.1 * (successes / mu - 0.5))
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 1e-4, 10.0)

            # Update mean
            mean = new_mean

            # Update C (simplified: use correlation of selected points)
            # For efficiency, we'll use a simplified covariance update
            if gen % 10 == 0:
                # Recompute C from selected points
                centered = np.array([s - mean for s in selected])
                C = np.cov(centered.T) + 1e-8 * np.eye(d)
                # Ensure positive semi-definite
                C = (C + C.T) / 2

            # Update paths (simplified)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (mean - x) / sigma
            x = mean

            fx = best_fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
