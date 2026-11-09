"""CMA-ES based optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        sigma = 2.0
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))  # population size
        c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 - (dim + 1) / (dim + 2)) / ((dim + 2) ** 2 + 20))
        c_sigma = (mu_sigma + 5) / ((dim + sigma) ** 2 + mu_sigma + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_sigma - 1) / (dim + 1)) - 1) + c_sigma
        mu_sigma = int(np.ceil(lambda_pop / 2))
        c_sigma = (mu_sigma + 5) / ((dim + sigma) ** 2 + mu_sigma + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_sigma - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        c_c = (4 + dim / dim) / (dim + 4 + 2 * dim / dim)
        d_c = 1 + 2 * max(0, np.sqrt((mu_sigma - 1) / (dim + 1)) - 1) + c_c

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        one_sixth = 1 / (4 * dim + 2)

        # Precompute weights
        weights = np.log(mu_sigma + 0.5) - np.log(np.arange(1, mu_sigma + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)

        calls_made = 1  # initial evaluation

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            # Use Cholesky decomposition of C for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C is not positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C)

            Y = np.array([sigma * (rng.multivariate_normal(np.zeros(dim), L)) for _ in range(lambda_pop)])
            Y = x[None, :] + Y

            # Evaluate population
            F = []
            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                y_i = np.clip(Y[i], -5, 5)
                f_i = func(y_i)
                F.append(f_i)
                calls_made += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = y_i.copy()

            if len(F) < lambda_pop:
                break

            # Sort by fitness
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]
            F_sorted = F[sorted_idx]

            # Update mean
            x_new = np.sum(weights[:, None] * Y_sorted[:mu_sigma], axis=0)
            
            # Update evolution paths
            y_diff = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lambda_pop)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff

            # Update covariance matrix
            if hsig:
                C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - c_c) * (1 - c_c) * (np.eye(dim) - C))
            else:
                C = (1 - c1 - cmu) * C + c1 * (1 - c_c) * (1 - c_c) * (np.eye(dim) - C)
            
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C += cmu * sum(w_i * np.outer(Y_sorted[i] - x, Y_sorted[i] - x) / sigma ** 2 for i, w_i in enumerate(weights))

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            x = x_new

            # Symmetrize C
            C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
