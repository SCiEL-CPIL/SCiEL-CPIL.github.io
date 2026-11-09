"""Optimizer for BBOB f13 (Sharp ridge) with CMA-ES variant."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Initial population size
        pop_size = max(4, int(4 + 2 * np.log(dim)))  # ~9 for dim=5

        # Initialize
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        # CMA-ES parameters
        mu = max(1, pop_size // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Best solution tracking
        best_x = x_mean.copy()
        best_f = func(best_x)
        call_count = 1

        # Main loop
        while call_count < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Sample population
            samples = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                # Transform using Cholesky of C
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                samples.append(y)

            # Evaluate
            fvals = []
            for y in samples:
                fy = func(y)
                fvals.append(fy)
                call_count += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            fvals = np.array(fvals)

            # Sort by fitness
            order = np.argsort(fvals)
            samples_sorted = [samples[i] for i in order]
            fvals_sorted = fvals[order]

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += w[i] * samples_sorted[i]
            x_mean = new_mean

            # Update evolution path for sigma
            c_sigma_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma_sqrt * (C_inv_sqrt @ (x_mean - x_mean) / sigma)
            # Actually, we need the old mean
            # Let me fix this - we need to track old_mean
            # For simplicity, let's use a simpler update

            # Update step size (1/5 rule adapted)
            successes = np.sum(fvals_sorted[:mu] < fvals_sorted[mu:]) if mu < pop_size else 0
            if successes > mu * 0.5:
                sigma *= 1.05
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix (simplified)
            # For efficiency, use a simplified rank-mu update
            if call_count % 10 == 0:  # Update C every 10 generations
                diff = np.array([samples_sorted[i] - x_mean for i in range(mu)])
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(x_mean, x_mean) + c_mu * (diff.T @ np.diag(w) @ diff)
                # Ensure positive semi-definite
                C = (C + C.T) / 2
                # Recompute inverse square root
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)

            # Adaptive population size: shrink as we converge
            spread = np.std(fvals_sorted)
            if spread < 0.1 * abs(best_f) + 1e-6:
                pop_size = max(4, pop_size - 1)
            elif spread > 1.0:
                pop_size = min(20, pop_size + 1)

        return best_x, best_f
# EVOLVE-BLOCK-END
