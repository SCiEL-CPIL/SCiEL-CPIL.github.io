"""CMA-ES based optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive sigma for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start at center of the box
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size

        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Cholesky factor of C
        C_inv_sqrt = np.eye(dim)  # C^(-1/2)
        B = np.eye(dim)  # Eigenvectors
        D = np.ones(dim)  # Eigenvalues

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
        mu = lambda_ // 2  # Number of recombination weights

        # Recombination weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / w.sum()

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Threshold for h_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        hs = (1 + c_sigma) ** (1 / (budget * c_sigma)) if budget > 0 else 1.0
        # Actually, the threshold is based on the expected norm of the evolution path
        # Let's use the standard formulation
        hs_threshold = (1 + 2 / (dim + 1)) * chi_n

        # Evolution path for sigma
        p_sigma = np.zeros(dim)
        # Evolution path for covariance
        p_c = np.zeros(dim)

        # Initial function evaluation
        fx = func(mean)
        best_x = mean.copy()
        best_fx = fx
        calls = 1

        # Precompute some constants
        # We need to be careful with the budget: we have 'budget' total calls
        # We've used 1, so we have budget - 1 left

        # To simplify and ensure we stay within budget, we'll do a simplified CMA-ES
        # that doesn't strictly follow all the update rules but captures the essence:
        # adapt covariance based on successful steps, adapt sigma based on success rate.

        # Actually, let's implement a more standard CMA-ES but with careful budget management.
        # Each generation uses lambda_ evaluations.
        # Number of generations we can afford: (budget - 1) // lambda_

        # Let's use a simpler but effective approach:
        # CMA-ES with standard updates, but we'll track calls carefully.

        # Reset for proper CMA-ES
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C_inv_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        fx = func(mean)
        best_x = mean.copy()
        best_fx = fx
        calls = 1

        # We'll run generations as long as we have budget
        while calls + lambda_ <= budget:
            # Generate population
            Y = np.zeros((lambda_, dim))
            X = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)

            for i in range(lambda_):
                # z ~ N(0, I)
                z = np.random.standard_normal(dim)
                # y = B @ (D * z)
                y = B @ (D * z)
                # x = mean + sigma * y
                x = mean + sigma * y
                # Clip to box
                x = np.clip(x, -5, 5)
                X[i] = x
                F[i] = func(x)
                calls += 1

                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = x.copy()

            # Sort by fitness
            idx = np.argsort(F)
            F_sorted = F[idx]
            X_sorted = X[idx]

            # Update mean (recombination)
            mean_old = mean.copy()
            mean = sum(w[i] * X_sorted[i] for i in range(mu))

            # Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (mean - mean_old) / sigma
            diff = (mean - mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff)

            # h_sigma: check if p_sigma is small enough
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (budget - calls))) < (1 + 2 / (dim + 1)) * chi_n else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))

            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt @ (mean - mean_old) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff)

            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (y_i @ y_i))
            # where y_i = (x_i - mean_old) / sigma

            # Compute y_i for the selected individuals
            Y_selected = np.zeros((mu, dim))
            for i in range(mu):
                Y_selected[i] = (X_sorted[i] - mean_old) / sigma

            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            for i in range(mu):
                C += c_mu * w[i] * np.outer(Y_selected[i], Y_selected[i])

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Eigendecomposition of C
            try:
                D_sq, B = np.linalg.eigh(C)
                # Clip eigenvalues to be positive
                D_sq = np.maximum(D_sq, 1e-12)
                D = np.sqrt(D_sq)
                C_inv_sqrt = (B / D) @ B.T
            except np.linalg.LinAlgError:
                # Fallback to identity if eigendecomposition fails
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                C_inv_sqrt = np.eye(dim)

            # Check for convergence
            if sigma < 1e-8:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
