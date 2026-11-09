"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_used = 0
        n = self.dim

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return None
            calls_used += 1
            return float(func(np.asarray(x, dtype=float)))

        # CMA-ES parameters
        pop_size = 4
        # Population size: lambda = 4
        # Standard CMA-ES parameters for lambda=4, n=5
        mu = max(1, pop_size // 2)  # number of recombination weights = 2
        # Weights for recombination
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(i + 1) - np.log(mu)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Mean update parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial mean: origin (minimizer is in [-4,4]^5)
        mean = np.zeros(n)
        f_mean = safe_call(mean)
        if f_mean is None:
            return mean, f_mean

        # Initial step size
        sigma = 1.0

        # Covariance matrix
        C = np.eye(n)
        # Precompute C^0.5
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Track best solution
        best_x = mean.copy()
        best_f = f_mean

        # Cache of evaluated points to avoid re-evaluation
        eval_cache = {}

        def get_f(x):
            key = tuple(np.round(x, 10))
            if key in eval_cache:
                return eval_cache[key]
            f = safe_call(x)
            if f is not None:
                eval_cache[key] = f
            return f

        # Main CMA-ES loop
        while calls_used < self.budget:
            if time.time() > deadline - 5.0:
                break

            # Sample population
            pop = []
            for i in range(pop_size):
                z = np.random.randn(n)
                x_i = mean + sigma * (C_sqrt @ z)
                pop.append(x_i)

            # Evaluate population
            pop_f = []
            for x_i in pop:
                f_i = get_f(x_i)
                if f_i is None:
                    f_i = np.inf
                pop_f.append(f_i)

            # Update best
            for i in range(pop_size):
                if pop_f[i] < best_f:
                    best_f = pop_f[i]
                    best_x = pop[i].copy()

            # Sort population by fitness
            idx = np.argsort(pop_f)
            pop = [pop[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]

            # Check if all evaluations failed
            if all(f == np.inf for f in pop_f):
                break

            # Recombination
            mean_old = mean.copy()
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * pop[i]
            mean = new_mean

            # Update evolution path for sigma
            z_mean = np.linalg.solve(C_sqrt, (mean - mean_old) / sigma) if sigma > 1e-10 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean

            # h_sigma: test if p_sigma is consistent with expected norm
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = np.sqrt(n) * (1 - (1 - c_sigma) ** 2)
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** 2)) < (1.4 + 2 / (n + 1)) * np.sqrt(n) else 0.0

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(n) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 50.0)  # Cap to avoid explosion

            # Update evolution path for C
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (pop[i] - mean_old) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)

            C = (1 + c_c * delta_h - c_c - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(n)) + c_mu * rank_mu

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Recompute C^0.5 using eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            C_sqrt_inv = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Check convergence: if sigma is very small, break
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
