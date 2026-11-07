"""Rewritten optimizer for BBOB f13 (Sharp ridge) with population-based CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 50, deadline guard, and budget enforcement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initialize
        rng = np.random.default_rng(42)
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        lambda_ = 50  # population size
        mu = lambda_ // 2  # number of parents
        # Weights (logarithmic)
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Step size
        sigma = 1.0

        # Covariance matrix
        C = np.eye(n)

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - 1 / mu_eff + 2 * mu_eff / (n + 2)) / ((n + 2) ** 2 + mu_eff)
        c_mu = (2 - c_1 + 2 * w[0] * mu_eff / (n + 2)) / ((n + 2) ** 2 + mu_eff)

        # Eigen decomposition cache
        D = np.ones(n)
        B = np.eye(n)

        def eval_point(xp):
            nonlocal calls, best_x, best_fx
            xp = np.clip(xp, -5, 5)
            fp = func(xp)
            calls += 1
            if fp < best_fx:
                best_fx = fp
                best_x = xp.copy()
            return xp, fp

        # Initial population evaluation
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            fitness = []

            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y, fy = eval_point(y)
                offspring.append(y)
                fitness.append(fy)

            if len(offspring) == 0:
                break

            # Sort by fitness
            idx = np.argsort(fitness)
            offspring = [offspring[i] for i in idx]
            fitness = [fitness[i] for i in idx]

            # Update mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += w[i] * offspring[i]
            x = x_new

            # Update evolution paths
            # p_sigma
            diff = (B @ (D * (B.T @ (x - x_new)))) / sigma if sigma > 0 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff

            # p_c
            diff_c = (x - x_new) / sigma if sigma > 0 else np.zeros(n)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff_c

            # Update covariance
            rank1 = np.outer(p_c, p_c)
            rankmu = np.zeros((n, n))
            for i in range(mu):
                y_i = (offspring[i] - x_new) / sigma if sigma > 0 else np.zeros(n)
                rankmu += w[i] * np.outer(y_i, y_i)

            C = (1 + c_1 * (mu_eff - 1) - c_1) * C + c_1 * rank1 + c_mu * rankmu

            # Update sigma
            p_sigma_norm = np.linalg.norm(p_sigma)
            chisigma = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chisigma - 1))

            # Clip sigma
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Periodic eigen decomposition
            if calls % 50 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip eigenvalues
                eigvals = np.clip(eigvals, 1e-12, 1e12)
                D = np.sqrt(eigvals)
                B = eigvecs

            # Check convergence
            if sigma < 1e-12:
                break

        return best_x, best_fx
