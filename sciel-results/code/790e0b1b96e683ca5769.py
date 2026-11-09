"""CMA-ES optimizer for BBOB-style f20 (Schwefel) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, integrated restart, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # CMA-ES parameters
        pop_size = 4
        # Initial sigma: explore the box
        sigma = 2.0

        # Means for recombination
        mu = pop_size // 2
        # Weights: standard CMA-ES weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, pop_size + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Chi-distribution expectation
        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))

        rng = np.random.default_rng(42)

        # Initial mean: center of the box
        mean = np.zeros(dim)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)  # C^{-1/2}
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1

        # Restart condition: stagnation detection
        no_improve = 0
        stagnation_limit = int(2 * (dim + 1) * np.log(dim + 1))

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(pop_size):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1

            # Check time mid-loop
            if time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Weighted recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += w[i] * ys_sorted[i]

            # Update step size (sigma)
            y_old = mean
            mean_old = mean
            mean = y_w

            # Update C^{-1/2} (approximation: use eigenvalue decomposition periodically)
            # For efficiency, we'll update C and recompute C^{-1/2} every few iterations
            
            # Path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ ((mean - mean_old) / sigma))

            # Path for C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ ((mean - mean_old) / sigma))

            # Update covariance matrix
            C = (1 + c_1 * (c_mu + c_1 - 1)) * C + c_1 * (p_c @ p_c - C)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - 1))) < (1.4 + 2 / (dim + 1)) * chi else 0.0
            C += c_mu * h_sigma * (p_c @ p_c - C)
            for i in range(mu):
                diff = C_inv_sqrt @ (ys_sorted[i] - mean_old) / sigma
                C += c_mu * w[i] * (diff @ diff - C)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 10)

            # Recompute C^{-1/2}
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Restart if stagnated too long
            if no_improve > stagnation_limit:
                # Restart with new initial point
                mean = rng.uniform(-4, 4, dim)
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma = 2.0
                no_improve = 0
                # Evaluate the new mean
                fm = func(mean)
                calls += 1
                if fm < best_f:
                    best_f = fm
                    best_x = mean.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
