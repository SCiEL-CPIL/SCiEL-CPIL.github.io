"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size and boundary handling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Deadline guard: return best found if we get close to the limit
        start_time = time.time()
        deadline = start_time + 280  # 280s out of 300s limit

        # Initial point: center of the box
        x_mean = np.zeros(dim)
        fx = func(x_mean)
        best_x = x_mean.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim)) for dim=5 -> 4+3*1.609=8.8 -> 9
        lam = 4 + int(3 * np.log(dim))
        # Step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for sampling
        D, B = self._eig_decompose(C)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / dim) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) * c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Counter for restarts (not really needed here but keep for structure)
        restarts = 0

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample population
            y_list = []
            x_list = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                y_list.append(y)
                x_candidate = np.clip(x_mean + sigma * y, -5, 5)
                x_list.append(x_candidate)

            # Evaluate population
            f_list = []
            for x_cand in x_list:
                f_val = func(x_cand)
                f_list.append(f_val)
                calls += 1

            # Sort by fitness
            indices = np.argsort(f_list)
            x_sorted = [x_list[i] for i in indices]
            f_sorted = [f_list[i] for i in indices]

            # Update best
            if f_sorted[0] < best_fx:
                best_fx = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Compute weighted mean of best mu solutions
            x_w = np.zeros(dim)
            for i in range(mu):
                x_w += weights[i] * x_sorted[i]

            # Update evolution paths
            y_w = (x_w - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1)) / chi_n < 1.4 + 2 / (dim + 1))
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

            # Update covariance matrix
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            outer = np.outer(p_c, p_c)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C
            C += c_1 * (outer + delta_h * C)
            for i in range(mu):
                y_i = (x_sorted[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Update step size
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x_mean = x_w

            # Re-decompose covariance matrix periodically
            if calls % lam == 0:
                D, B = self._eig_decompose(C)

            # Check for convergence
            if sigma < 1e-12 or np.all(np.abs(x_mean) < 1e-14):
                break

        return best_x, best_fx

    def _eig_decompose(self, C):
        """Eigen decomposition of C for sampling."""
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Clip negative eigenvalues to zero
        eigenvalues = np.maximum(eigenvalues, 0)
        D = np.sqrt(eigenvalues)
        B = eigenvectors
        return D, B
# EVOLVE-BLOCK-END
