"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic CMA-ES variant with Latin Hypercube initialization and adaptive parameters."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n_calls = 0

        # Latin Hypercube sampling for initialization - use multiple points
        # to find a better starting point
        n_init = min(10, max(5, budget // 50))
        golden = 0.6180339887498949
        init_points = np.zeros((n_init, dim))
        for i in range(n_init):
            for d in range(dim):
                # Use stratified sampling with golden ratio offsets
                init_points[i, d] = -5 + 10 * (((i + d * 0.618 + 0.5) % 1.0) / n_init + 0.5 * (1/n_init))
        
        init_points = np.clip(init_points, -5, 5)
        
        # Evaluate initial points
        best_init_val = np.inf
        best_init_point = init_points[0].copy()
        for i in range(n_init):
            if n_calls >= budget:
                break
            val = func(init_points[i])
            n_calls += 1
            if val < best_init_val:
                best_init_val = val
                best_init_point = init_points[i].copy()
        
        fx = best_init_val
        x = best_init_point.copy()

        # CMA-ES parameters
        # Adaptive population size: larger early, smaller later
        lam_base = max(12, 4 + 3 * int(np.log(dim + 1)))
        mu = lam_base // 2

        # Step size - start with larger scale for better exploration
        sigma = 2.0

        # Covariance matrix (identity initially)
        C = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Weights - use more aggressive selection
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1. / np.sum(weights**2)

        # Learning rates
        c_sigma = (mu_eff + 0.5) / (dim + 2)
        c_c = (4 - mu_eff + c_sigma) / ((dim + 1.3) * (dim + mu_eff))
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2.)**2 + mu_eff))

        # Damping
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1 / (21 * dim**2))

        # Main CMA-ES loop
        gen = 0
        while n_calls < budget:
            # Adaptive lambda: use more evaluations when budget is large
            remaining = budget - n_calls
            if remaining > 100:
                lam = lam_base
            elif remaining > 50:
                lam = max(8, lam_base - 2)
            else:
                lam = max(6, lam_base - 4)
            
            mu_adapt = max(3, lam // 2)
            
            # Recompute weights for adaptive mu
            w = np.log(mu_adapt + 0.5) - np.log(np.arange(1, mu_adapt + 1))
            w /= np.sum(w)
            mu_eff_adapt = 1. / np.sum(w**2)

            # Recompute learning rates for adaptive mu
            c_sigma_adapt = (mu_eff_adapt + 0.5) / (dim + 2)
            c_c_adapt = (4 - mu_eff_adapt + c_sigma_adapt) / ((dim + 1.3) * (dim + mu_eff_adapt))
            c_1_adapt = 2 / ((dim + 1.3)**2 + mu_eff_adapt)
            c_mu_adapt = min(1 - c_1_adapt, 2 * (mu_eff_adapt - 2 + 1. / mu_eff_adapt) / ((dim + 2.)**2 + mu_eff_adapt))

            # Eigendecomposition of C
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-14)
            B = evecs
            D = np.sqrt(evals)

            # Sample population
            # Use deterministic seed based on generation for reproducibility
            rng = np.random.default_rng(42 + gen * 7)
            Y = rng.standard_normal((lam, dim))

            # Transform to get X
            X = np.tile(x, (lam, 1)) + sigma * (Y @ (B * D))

            # Reflect points that hit boundaries instead of clipping
            for i in range(lam):
                for d in range(dim):
                    if X[i, d] < -5:
                        X[i, d] = -5 + (-5 - X[i, d]) * 0.5
                    elif X[i, d] > 5:
                        X[i, d] = 5 - (X[i, d] - 5) * 0.5
                    X[i, d] = np.clip(X[i, d], -5, 5)

            # Evaluate population
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                F[i] = func(X[i])
                n_calls += 1

            # If we ran out of budget, update best and break
            if n_calls >= budget:
                for i in range(lam):
                    if F[i] < fx:
                        fx = F[i]
                        x = X[i].copy()
                break

            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]

            # Update best
            if F_sorted[0] < fx:
                fx = F_sorted[0]
                x = X_sorted[0].copy()

            # Recombination
            m = np.sum(w[:, np.newaxis] * X_sorted[:mu_adapt], axis=0)

            # Update evolution path for sigma
            z = (m - x) / sigma
            p_sigma = (1 - c_sigma_adapt) * p_sigma + np.sqrt(c_sigma_adapt * (2 - c_sigma_adapt) * mu_eff_adapt) * (B @ (D @ z))

            # Hessian update
            norm_p_sigma = np.linalg.norm(p_sigma)
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            h_sigma = 1 if norm_p_sigma / np.sqrt(1 - (1 - c_sigma_adapt)**(gen + 1)) < threshold else 0

            # Update evolution path for C
            p_c = (1 - c_c_adapt) * p_c + h_sigma * np.sqrt(c_c_adapt * (2 - c_c_adapt) * mu_eff_adapt) * (B @ (D @ z))

            # Update covariance matrix using rank-one and rank-mu updates
            delta_h = (h_sigma - 1) * c_c_adapt * (2 - c_c_adapt)
            C = (1 + c_1_adapt * delta_h - c_1_adapt - c_mu_adapt) * C + \
                c_1_adapt * np.outer(p_c, p_c) + \
                c_mu_adapt * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(X_sorted[:mu_adapt] - x, X_sorted[:mu_adapt] - x) / sigma**2, axis=0)

            # Symmetrize and regularize for numerical stability
            C = (C + C.T) / 2
            # Add small regularization to prevent singularity
            C += 1e-12 * np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma_adapt / chi_n * (norm_p_sigma / chi_n - 1))

            # Clamp sigma to prevent numerical issues
            sigma = np.clip(sigma, 1e-8, 1e4)

            # Update mean
            x = m

            gen += 1

        return x, fx
# EVOLVE-BLOCK-END
