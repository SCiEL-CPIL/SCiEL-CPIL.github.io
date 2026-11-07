"""Optimizer for BBOB f11 (Discus) using CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin below 330s

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(d)))
        sigma = 2.0

        # Initial point at origin (good for symmetric box)
        x = np.zeros(d)
        fx = func(x)

        # Initial covariance (identity)
        C = np.eye(d)
        C_eigenvalues = np.ones(d)
        C_eigenvectors = np.eye(d)

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)

        # Learning rates
        cs = (lambda_pop + 2) / (d + lambda_pop + 5)
        cmu = 2 / ((d + 2.) ** 2 + lambda_pop)
        cc = (4 + lambda_pop / d) / (d + 4 + 2 * lambda_pop / d)
        c1 = (2 + 2.0 / (d + 1.3)) / ((d + 1) ** 2 + lambda_pop)
        cmu = min(cmu, c1 * (3 - c1) / 2)

        # Damping for step size adaptation
        dsigma = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (d + 1)) - 1) + cs
        # chi_n: expected norm of standard normal
        chi_n = np.sqrt(d) * (1 - 1. / (4 * d) + 1. / (21 * d ** 2))

        # Counters
        count_steps = 0
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Precompute initial inverse sqrt of C
        def _update_eigen():
            nonlocal C_eigenvalues, C_eigenvectors
            C_eigenvalues, C_eigenvectors = np.linalg.eigh(C)
            C_eigenvalues = np.maximum(C_eigenvalues, 1e-12)

        _update_eigen()

        while calls < budget:
            # Time guard
            if time.time() > deadline:
                break

            # Generate population
            pop = np.zeros((lambda_pop, d))
            pop_f = np.full(lambda_pop, np.inf)

            # Inverse square root of C via eigen decomposition
            sqrt_inv_C = C_eigenvectors @ np.diag(1.0 / np.sqrt(C_eigenvalues)) @ C_eigenvectors.T

            for i in range(lambda_pop):
                zi = np.random.standard_normal(d)
                yi = x + sigma * (sqrt_inv_C @ zi)
                yi = np.clip(yi, -5, 5)
                pop[i] = yi
                pop_f[i] = func(yi)
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            if calls >= budget:
                break

            # Update best
            for i in range(lambda_pop):
                if pop_f[i] < best_fx:
                    best_fx = pop_f[i]
                    best_x = pop[i].copy()

            # Sort population by fitness
            order = np.argsort(pop_f)
            pop_sorted = pop[order]
            pop_f_sorted = pop_f[order]

            # Weighted recombination
            weights = np.log(lambda_pop / 2 + 0.5) - np.log(np.arange(1, lambda_pop + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights[:min(lambda_pop // 2, d + 1)] ** 2)

            # Weighted mean of top mu individuals
            mu = min(int(mueff), lambda_pop)
            y_mean = np.zeros(d)
            for i in range(mu):
                y_mean += weights[i] * (pop_sorted[i] - x)
            y_mean /= np.sum(weights[:mu])

            # Update evolution paths
            inv_sqrt_C = C_eigenvectors @ np.diag(1.0 / np.sqrt(C_eigenvalues)) @ C_eigenvectors.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ (y_mean / sigma))
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1)) / chi_n < (1.4 + 2 / (d + 1)) * cc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_mean / sigma

            # Update covariance matrix
            delta_hs = (1 - hsig) * cc * (2 - cc)
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((d, d))
            for i in range(mu):
                zi = (pop_sorted[i] - x) / sigma
                rankmu += weights[i] * np.outer(zi, zi)

            C = (1 + c1 * delta_hs - c1 - cmu * np.sum(weights[:mu])) * C \
                + c1 * rank1 + cmu * rankmu

            # Step size adaptation
            sigma *= np.exp((cs / dsigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e6)

            # Update mean
            x = x + y_mean
            x = np.clip(x, -5, 5)

            # Update eigen decomposition periodically
            if calls % lambda_pop == 0:
                _update_eigen()

            # If convergence is too slow, nudge sigma
            if sigma < 1e-6:
                break

        return best_x, best_fx
