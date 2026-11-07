"""CMA-ES optimizer for BBOB f17 (Schaffers F7) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for low-dimensional black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(n)
        f = func(x)

        # CMA-ES parameters
        sigma = 1.0
        lambda_ = 4 * n  # population size
        mu = lambda_ // 2  # recombination weight count

        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # State
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)  # We'll track via eigen decomposition or use identity initially
        # Actually, for simplicity and speed in low dim, let's use a simpler update
        # But CMA-ES needs C^{-1/2}. Let's use Cholesky or eigen each step.
        # For n=5, this is cheap.

        # Initial eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # Path length normalization
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Best solution tracking
        best_x = x.copy()
        best_f = f

        # Main loop
        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline:
                break

            # Generate lambda samples
            samples = []
            for i in range(lambda_):
                z = np.random.randn(n)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                samples.append(y)

            # Evaluate
            fitness = []
            for y in samples:
                if time.time() > deadline:
                    fitness.append(np.inf)
                else:
                    fitness.append(func(y))

            # Sort by fitness
            idx = np.argsort(fitness)
            samples = [samples[i] for i in idx]
            fitness = [fitness[i] for i in idx]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = samples[0].copy()

            # Recombination
            m_new = np.zeros(n)
            for i in range(mu):
                m_new += weights[i] * samples[i]

            # Update step size control variable
            d_sigma = (1 + c_sigma) * (1 + 2 * max(0, (c_c + 1) / (c_c - 1) - 1))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ ((m_new - x) / sigma)
            norm_p_sigma = np.linalg.norm(p_sigma)

            # Update evolution path
            p_c = (1 - c_c) * p_c + c_c * (m_new - x) / sigma
            norm_p_c = np.linalg.norm(p_c)

            # Update covariance matrix
            delta_h = (1.7 + 2 / (n + 1)) * (1 - c_c)
            h = 1 if norm_p_c / np.sqrt(1 - (1 - c_c) ** (gen + 1)) < (1 + delta_h) * chi_n else 0

            # Rank-one update
            C += c_1 * (np.outer(p_c, p_c) / c_c * (2 - h) - (1 - c_1) * np.eye(n) * h)

            # Rank-mu update
            y_w = np.array([(m_new - x) / sigma for _ in range(mu)])
            C += c_mu * np.sum(weights[i] * np.outer(y_w[i], y_w[i]) for i in range(mu))

            # Ensure positive semi-definite
            C = (C + C.T) / 2

            # Update eigen decomposition for C^{-1/2}
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            except:
                pass

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Update center
            x = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
