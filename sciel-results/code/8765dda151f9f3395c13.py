"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like adaptive covariance matrix evolution strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 280s safety margin before 300s limit

        # Initialize CMA-ES parameters
        n = dim
        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1) ** 2 + c_sigma) / (
            (n + 2) + mu_eff - mu_eff / (n + 1) ** 2 + c_sigma
        )
        c_mu = (
            2 * (mu_eff / (n + 2) - 1) + c_1
        ) / (
            (n + 2 + c_sigma) + mu_eff - mu_eff / (n + 1) ** 2 + c_sigma
        )

        # Initial step size
        sigma = 2.0
        # Covariance matrix (start with identity)
        C = np.eye(n)
        # Cholesky factor of C
        C_inv_sqrt = np.eye(n)
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        # Cumulated step length for sigma adaptation
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Track best
        call_count = 1

        for iteration in range(budget):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lambda_):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sample from N(0, C)
                y = C_inv_sqrt @ rng_standard_normal(n) if hasattr(self, '_rng') else C_inv_sqrt @ np.random.standard_normal(n)

                # Actually, let's use a proper RNG
                # We need to maintain a random state
                if not hasattr(self, '_rng'):
                    self._rng = np.random.default_rng(42)

                y = C_inv_sqrt @ self._rng.standard_normal(n)
                x_trial = np.clip(x + sigma * y, -5, 5)
                f_trial = func(x_trial)
                call_count += 1

                ys.append(y)
                xs.append(x_trial)
                fs.append(f_trial)

                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()

            if len(xs) == 0:
                break

            # Sort by fitness
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            xs = [xs[i] for i in indices]
            fs = [fs[i] for i in indices]

            # Update mean
            y_w = sum(weights[i] * ys[i] for i in range(mu))
            x = np.clip(x + y_w, -5, 5)

            # Update sigma (CSA)
            p_sigma = (
                (1 - c_sigma) * p_sigma
                + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
                * C_inv_sqrt @ y_w / sigma
            )
            sigma *= np.exp(
                c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1)
            )
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C (CMU)
            p_c = (
                (1 - c_c) * p_c
                + c_c * (np.linalg.norm(p_sigma) / chi_n > 1.4 + 2 / (n + 1))
                * y_w / sigma
            )

            # Update covariance
            C = (
                (1 + c_1 * (2 - c_1) - c_mu) * C
                + c_1 * np.outer(p_c, p_c)
                + c_mu * sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            )

            # Re-estimate Cholesky factor periodically to maintain numerical stability
            if iteration % 10 == 0:
                try:
                    # Symmetrize
                    C = (C + C.T) / 2
                    # Add small regularization
                    C += 1e-12 * np.eye(n)
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    # Reset to identity if Cholesky fails
                    C = np.eye(n)
                    C_inv_sqrt = np.eye(n)
                    sigma = max(sigma, 0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
