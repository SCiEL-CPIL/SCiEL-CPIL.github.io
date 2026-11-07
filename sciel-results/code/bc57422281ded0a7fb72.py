"""Rewritten optimizer: CMA-ES with time guard for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s kill
        n = self.dim
        budget = self.budget

        # Initialize population size
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        
        # CMA-ES parameters
        sigma = 2.0
        c_sigma = (mu + 1) / (4 * n + 4)
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + mu)
        c_mu = min(1 - c_1, (2 + (mu / n - 1) ** 2) / ((n + 2) ** 2 + mu))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Random start point
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES state
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        for iteration in range(budget // lambda_):
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_, n))
            for i in range(lambda_):
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(n)
                pop[i] = x + sigma * (B @ (D * z))
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fvals = np.array([func(pop[i]) for i in range(lambda_)])

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Weighted mean
            m = np.average(pop_sorted[:mu], axis=0, weights=weights)

            # Update CMA-ES state
            y = (m - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) / chi_n) < 1.4 + 2 / (n + 1)
            if hsig:
                pc = (1 - c_c) * pc + c_c * y
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * np.eye(n)) \
                + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop_sorted[:mu] - x, pop_sorted[:mu] - x) / sigma**2, axis=0)

            # Eigendecomposition for stable update
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            C = B @ np.diag(D ** 2) @ B.T

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = m

        return best_x, best_fx
# EVOLVE-BLOCK-END
