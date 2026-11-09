"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box optimization on the Weierstrass function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit

        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of selected points
        lambda_pop = 2 * mu    # population size
        # Step size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 2.5 + mu)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1. / mu) / ((dim + 2.) ** 2 + mu))
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

        # Initial mean - start at center
        mean = np.zeros(dim)
        
        # Evaluate initial point
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        # Generate initial population
        def sample_population():
            nonlocal sigma, C, mean
            # Whitening
            B, D = np.linalg.eigh(C)
            B = B[:, D > 1e-10]
            D = D[D > 1e-10]
            # Sample
            z = np.random.default_rng(np.random.randint(10**6)).standard_normal((lambda_pop, dim))
            x = np.array([mean + sigma * (B @ (D ** 0.5 * z[i])) for i in range(lambda_pop)])
            x = np.clip(x, -5, 5)
            return x

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sample population
            rng = np.random.default_rng(calls * 7919 + 13)
            z = rng.standard_normal((lambda_pop, dim))
            B, D = np.linalg.eigh(C)
            D = np.maximum(D, 1e-10)
            x_pop = np.array([
                np.clip(mean + sigma * (B @ (np.sqrt(D) * z[i])), -5, 5)
                for i in range(lambda_pop)
            ])
            
            # Evaluate
            f_pop = np.array([func(x_pop[i]) for i in range(lambda_pop)])
            calls += lambda_pop
            
            # Update best
            idx_best = np.argmin(f_pop)
            if f_pop[idx_best] < best_f:
                best_f = f_pop[idx_best]
                best_x = x_pop[idx_best].copy()
            
            # Sort by fitness
            order = np.argsort(f_pop)
            x_w = x_pop[order[:mu]]
            f_w = f_pop[order[:mu]]
            
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum(weights[:, None] * x_w, axis=0)
            
            # Update step size
            y = (mean - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1)) / chi_n) < 1.4 + 2 / (dim + 1)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update covariance
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            C += c_mu * np.sum(weights[:, None, None] * np.outer(x_w - old_mean, x_w - old_mean) / sigma ** 2, axis=0)
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Eigenvalue correction
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
