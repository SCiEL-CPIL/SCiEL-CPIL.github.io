"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: center of the box (good for sphere with shift in [-4,4])
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        # Population size: standard is 4 + floor(3 * ln(n))
        lam = max(2, 4 + int(3 * np.log(dim)))
        # Ensure we don't exceed budget
        lam = min(lam, budget - 1)

        # Initial sigma
        sigma = 2.0

        # CMA-ES state
        # Mean
        mean = x.copy()
        # Covariance matrix (initially identity)
        C = np.eye(dim)
        # Step size
        sigma = 2.0
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Adaptation rates
        mu = lam // 2  # number of best individuals used for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Precompute sqrt of C for efficient sampling (will update)
        # For simplicity, we'll use Cholesky each iteration or update via rank updates
        # For dim=5, Cholesky is cheap

        def sample_from_normal():
            """Sample from N(0, C) using current sigma and C."""
            # Use Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C[:] = np.eye(dim)
                L = np.eye(dim)
            z = rng.standard_normal(dim)
            return sigma * (L @ z)

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            offspring_f = np.zeros(lam)

            for i in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sample
                z = sample_from_normal()
                y = mean + z
                # Clip to box
                y = np.clip(y, -5, 5)
                offspring[i] = y
                offspring_f[i] = func(y)
                calls += 1

                # Update best
                if offspring_f[i] < best_fx:
                    best_fx = offspring_f[i]
                    best_x = y.copy()

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = offspring[idx]
            offspring_f = offspring_f[idx]

            # Update mean
            new_mean = np.sum(weights[:, None] * offspring[:mu], axis=0)

            # Update evolution paths
            # ps update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C @ (new_mean - mean))

            # pc update
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam))) < (1.4 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma)):
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (new_mean - mean)

            # Update covariance
            rank_one = np.outer(new_mean - mean, new_mean - mean)
            rank_mu = np.sum(weights[:, None, None] * np.outer(offspring[:mu] - mean, offspring[:mu] - mean), axis=0)

            h_sigma = 1.0  # approximation
            C = (1 + c_1 * (h_sigma - 1) + c_mu) * C - c_1 * pc * pc.T - c_mu * rank_mu + c_1 * (h_sigma - 1) * np.outer(mean - mean, mean - mean)
            # Simplify: the last term is zero
            C = (1 + c_1 * (h_sigma - 1) + c_mu) * C - c_1 * pc * pc.T - c_mu * rank_mu

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(c_sigma * (2 - c_sigma)) - 1))

            # Update mean
            mean = new_mean

            # Ensure C is positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C = C + 1e-8 * np.eye(dim)

        return best_x, best_fx
# EVOLVE-BLOCK-END
