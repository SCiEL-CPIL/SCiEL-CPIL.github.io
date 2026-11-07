"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        rng = np.random.default_rng(42)
        n = self.dim

        # Initialize population
        pop_size = max(4, int(4 + 3 * np.log(n)))
        x_mean = rng.uniform(-3, 3, n)
        sigma = 2.0

        # Generate initial population
        pop = np.array([np.clip(x_mean + sigma * rng.standard_normal(n), -5, 5) for _ in range(pop_size)])
        fits = np.array([func(x) for x in pop])
        calls = pop_size

        # Track best
        best_idx = np.argmin(fits)
        x_best = pop[best_idx].copy()
        f_best = fits[best_idx]

        # CMA-ES parameters
        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Evolution path for covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (1 - c_sigma) + 2 * (1 - c_c) * c_sigma
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (n + 2) ** 2 - 1) * (1 - c_sigma))

        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Main loop
        while calls < self.budget:
            # Check time limit
            if time.time() > deadline - 2:
                break

            # Sort population by fitness
            order = np.argsort(fits)
            pop_sorted = pop[order]
            fits_sorted = fits[order]

            # Save old mean
            old_mean = x_mean.copy()

            # Update mean
            x_mean = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)
            x_mean = np.clip(x_mean, -5, 5)

            # Update best
            if fits_sorted[0] < f_best:
                f_best = fits_sorted[0]
                x_best = pop_sorted[0].copy()

            # Compute step
            step = (x_mean - old_mean) / sigma

            # Compute inverse square root of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update evolution path for covariance matrix adaptation
            pc = (1 - c_c) * pc + c_c * np.dot(inv_sqrt_C, step)

            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.dot(inv_sqrt_C, step)

            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix
            delta_sig = (1 + 2 * max(0, (np.linalg.norm(ps) / chi_n - 1) ** 0.5 - 1)) * c_sigma
            rank1 = np.outer(pc, pc)
            rankmu = np.sum(weights[:, np.newaxis] * np.outer(pop_sorted[:mu] - old_mean, pop_sorted[:mu] - old_mean) / sigma ** 2, axis=0)
            C = (1 - c_1 - c_mu + delta_sig) * C + c_1 * rank1 + c_mu * rankmu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Generate new population
            pop = []
            for _ in range(pop_size):
                z = rng.standard_normal(n)
                x = x_mean + sigma * np.dot(np.linalg.cholesky(C), z)
                x = np.clip(x, -5, 5)
                pop.append(x)
            pop = np.array(pop)
            fits = np.array([func(x) for x in pop])
            calls += pop_size

            # Check for convergence
            if sigma < 1e-10 or np.all(np.abs(fits_sorted[:5] - fits_sorted[0]) < 1e-10):
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
