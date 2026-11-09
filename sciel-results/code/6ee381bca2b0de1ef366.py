"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class Optimizer:
    """
    CMA-ES based optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
      - Use CMA-ES with a population size of 10.
      - Start from the origin (0,0,0,0,0).
      - Adapt the covariance matrix to handle the ill-conditioning.
      - Use a time guard to stay within the 330s limit.
      - Track the best point and value found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin

        dim = self.dim
        n_evals = 0
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        pop_size = 10
        sigma = 2.0  # Initial step size
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        # Step size control parameters
        cs = 1 / (pop_size + 4)
        ds = 1 + 2 * max(0, (pop_size / 2 - 1) / (dim + 2)) + cs
        # Covariance matrix adaptation parameters
        cmu = 2 / ((dim + 1.3)**2 + pop_size)
        # Path parameters
        path_psig = np.zeros(dim)
        path_pc = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = func(best_x)
        n_evals += 1

        # Weights for recombination
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        # Learning rates
        c_sigma = (mueff + 2) / ((dim + mueff) * (dim + 2))
        c_c = (4 + mueff / dim) / ((dim + 4) * (dim + mueff / 2))
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        while n_evals < self.budget:
            if time.time() > deadline:
                break

            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                L = np.eye(dim)

            z = rng.standard_normal((pop_size, dim))
            x_pop = np.array([x_mean + sigma * (z[i] @ L) for i in range(pop_size)])
            
            # Evaluate population
            f_pop = np.zeros(pop_size)
            for i in range(pop_size):
                if n_evals >= self.budget:
                    break
                f_pop[i] = func(x_pop[i])
                n_evals += 1
            
            if n_evals >= self.budget:
                break

            # Sort by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Update mean
            x_mean = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)

            # Update path p_sigma
            path_psig = (1 - cs) * path_psig + cs * np.sqrt(mueff) * L.T @ (x_mean - x_mean_prev) / sigma
            # Note: x_mean_prev is the mean from the previous iteration
            # We need to store the previous mean
            if not hasattr(self, '_prev_mean'):
                self._prev_mean = x_mean
            else:
                x_mean_prev = self._prev_mean
                self._prev_mean = x_mean

            # Correction for the path p_sigma
            path_psig = (1 - cs) * path_psig + cs * np.sqrt(mueff) * np.linalg.inv(L) @ (x_mean - x_mean_prev) / sigma

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(path_psig) / chi_n - 1))

            # Update path p_c
            h_sigma = 1 if np.linalg.norm(path_psig) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            path_pc = (1 - cc) * path_pc + h_sigma * cc * (x_mean - x_mean_prev) / sigma

            # Update covariance matrix C
            delta_h = (2 - h_sigma) * c_c - 1
            C = (1 + c_1 + c_mu - c_c - delta_h) * C + \
                c_1 * np.outer(path_pc, path_pc) + \
                c_mu / mueff * np.sum(weights[i] * np.outer(x_sorted[i] - x_mean_prev, x_sorted[i] - x_mean_prev) for i in range(mu))

            # Ensure C is positive semi-definite
            # Add a small diagonal if needed
            C += 1e-10 * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
