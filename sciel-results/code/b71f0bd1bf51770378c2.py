"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time and budget guards for BBOB optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        # CMA-ES parameters for n=5
        lam = 10  # Population size
        mu = lam // 2  # Number of selected points
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1) ** 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff / (n + 1) ** 2 - 1) + c_1) / ((n + 2) + mu_eff / (n + 1) ** 2 - 2)

        # Initial point and step size
        x = np.zeros(n)
        sigma = 2.0

        # Covariance matrix and its inverse
        C = np.eye(n)
        C_inv = np.eye(n)
        # Cholesky decomposition for sampling
        L = np.eye(n)

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Track best solution
        best_x = x.copy()
        best_fx = float('inf')

        calls_made = 0

        def eval_point(pt):
            nonlocal calls_made, best_x, best_fx
            calls_made += 1
            val = func(pt)
            if val < best_fx:
                best_fx = val
                best_x = pt.copy()
            return val

        # Initial evaluation
        fx_init = eval_point(x)
        best_fx = fx_init
        best_x = x.copy()

        # Precompute eigenvalues for C
        eigvals = np.ones(n)

        # Main loop
        gen = 0
        while calls_made < budget:
            # Check time
            if time.time() - start_time > time_limit - 1.0:
                break

            # Sample lambda points
            population = []
            for i in range(lam):
                # Sample from multivariate normal using Cholesky
                z = np.linalg.cholesky(C) @ np.random.standard_normal(n)
                y = x + sigma * z
                population.append(y)

            # Evaluate population
            fvals = np.array([eval_point(p) for p in population])

            # Sort by fitness
            idx = np.argsort(fvals)
            y_sorted = [population[i] for i in idx]
            f_sorted = fvals[idx]

            # Update mean
            x_old = x.copy()
            x = sum(w[i] * y_sorted[i] for i in range(mu))

            # Update sigma
            # Calculate norm of p_sigma
            p_sigma = ((1 - c_sigma) * p_sigma +
                       np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma *
                       np.linalg.cholesky(C) @
                       (x - x_old) / sigma)

            # Check for step size adaptation
            # Approximate log expectation of chi distribution
            chi_n = np.sqrt(n * (1 - 1 / (4 * n) - 1 / (25 * n ** 2)))
            if np.linalg.norm(p_sigma) / chi_n < (1.4 + 2 / (n + 1)) * (1 - c_sigma):
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

            # Update covariance matrix
            # Update evolution path p_c
            p_c = ((1 - c_c) * p_c +
                   np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma)

            # Compute delta_h
            delta_h = 1 if (np.linalg.norm(p_c) ** 2 / (1 - (1 - c_c) ** (2 * (gen + 1)))
                           < (2 + 2 / (n + 1)) * (1 - (1 - c_c) ** (2 * (gen + 1)))) else 0

            # Update C
            outer = np.zeros((n, n))
            for i in range(mu):
                outer += w[i] * np.outer(y_sorted[i] - x_old, y_sorted[i] - x_old)
            C = ((1 - c_1 - c_mu) * C +
                 c_1 * (C + np.outer(p_c, p_c) - p_c @ C - C @ p_c.T + delta_h * np.outer(p_c, p_c)) +
                 c_mu * outer)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                C_inv = np.linalg.inv(C)
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: add small regularization
                C += 1e-6 * np.eye(n)
                C = (C + C.T) / 2
                try:
                    C_inv = np.linalg.inv(C)
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset C to identity
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    L = np.eye(n)

            gen += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
