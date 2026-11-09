<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        return x, fx
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))
        mu = lam // 2

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / d) / ((d + 2) ** 2 + mu_eff))

        chi_d = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        # Initial state
        x_mean = rng.uniform(-4, 4, d)
        sigma = 2.0
        B = np.eye(d)
        D = np.ones(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1

        # Precompute D^-1 B^T for efficiency
        def step():
            nonlocal B, D, sigma
            # Generate population
            z = rng.standard_normal((lam, d))
            # Transform: y = x_mean + sigma * B @ (D * z)
            pop = np.einsum('ij,j,ji->ki', B, D, z) * sigma + x_mean
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fvals = np.array([func(x) for x in pop])
            calls += lam

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Weighted mean of best mu
            y_w = np.sum(weights[:, None] * pop_sorted[:mu], axis=0)

            # CMA update
            B_inv_D_inv = np.linalg.inv(B.T)  # This is expensive; let's use a different approach
            # Actually, let's use the standard CMA-ES formulation more carefully

            # For simplicity and speed, use a simplified CMA-ES
            # Update mean
            x_mean_new = x_mean + c_1 * (y_w - x_mean)

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B.T @ (D * (y_w - x_mean)))
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1 + 2 / (d + 1)) * chi_d else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x_mean) / sigma

            # Covariance matrix update
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c + delta_h * x_mean, p_c + delta_h * x_mean)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                zi = (pop_sorted[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (2 * c_1 / c_1 - 1) * np.eye(d)) + c_mu * rank_mu

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition for B and D
            D2, B2 = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-10))
            D = D2
            B = B2

            # Step size adaptation (CSA)
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
            sigma = np.clip(sigma, 1e-6, 100)

            x_mean = x_mean_new
            x_mean = np.clip(x_mean, -5, 5)

            return calls

        # Initialize C
        C = np.eye(d)

        while calls < budget:
            if time.time() > deadline:
                break
            step()

        return best_x, best_f
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the above CMA-ES implementation has issues with the covariance matrix update and eigen decomposition. Let me write a cleaner, more correct implementation:

