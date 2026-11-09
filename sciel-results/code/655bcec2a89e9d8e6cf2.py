# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size for f13 (sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin under 300s
        n = self.dim
        budget = self.budget

        # Initialize CMA-ES
        x = np.zeros(n)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1

        sigma = 3.0  # initial step size (box is [-5,5], minimizer in [-4,4])
        C = np.eye(n)
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        # Parameters
        mu = max(1, n // 2)  # number of selected parents
        lambda_ = max(2 * n, 4)  # population size
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        stagnation_count = 0
        no_improve = 0
        best_f_prev = best_f

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Adaptive population size: shrink as we converge
            if no_improve > 20:
                lambda_ = max(2 * n, lambda_ - 1)
            if no_improve > 50:
                lambda_ = max(2 * n, lambda_ - 2)

            # Sample population
            offspring = np.zeros((lambda_, n))
            f_offspring = np.zeros(lambda_)
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                z = np.random.standard_normal(n)
                y = x + sigma * (z @ np.linalg.cholesky(C).T)
                y = np.clip(y, -5, 5)
                offspring[i] = y
                f_offspring[i] = func(y)
                calls += 1
                if f_offspring[i] < best_f:
                    best_f = f_offspring[i]
                    best_x = y.copy()

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(f_offspring)
            # Select top mu
            selected = offspring[order[:mu]]
            selected_f = f_offspring[order[:mu]]

            # Check for improvement
            if best_f < best_f_prev - 1e-10:
                no_improve = 0
                stagnation_count = 0
            else:
                no_improve += 1

            best_f_prev = best_f

            # Weighted mean
            m = np.sum(weights[:, None] * selected, axis=0)

            # Update evolution paths
            # Inverse square root of C for the step
            # Use eigendecomposition for stability
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-20)
            C_inv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (C_inv_sqrt @ (m - x) / sigma)
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls - budget))) \
                      / chi_n < (1.4 + 2 / (n + 1))

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (m - x) / sigma)

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                + c_1 * (p_c @ p_c - C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(selected - x, selected - x) / sigma ** 2, axis=0)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-15)
            C = eigvec @ np.diag(eigval) @ eigvec.T

            # Update mean
            x = m

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Evaluate new mean
            f_new = func(x)
            calls += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x.copy()

            # Stagnation restart: if stuck for a long time, add a small random perturbation
            if no_improve > 80:
                x = x + 0.5 * sigma * np.random.standard_normal(n)
                x = np.clip(x, -5, 5)
                sigma *= 1.5
                no_improve = 0

            # If sigma is very small, try a larger jump
            if sigma < 1e-6 and no_improve > 10:
                x = x + 1.0 * np.random.standard_normal(n)
                x = np.clip(x, -5, 5)
                sigma = 1.0
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
