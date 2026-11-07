"""Rewritten optimizer using CMA-ES style covariance adaptation with population."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES variant for BBOB-style f18 (Schaffers F7 ill-conditioned)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Parameters for CMA-ES
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # State variables
        x_mean = np.zeros(d)
        sigma = 5.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(d)
        gen = 0
        
        # Helper to evaluate and track
        def safe_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Initial evaluation
        safe_func(x_mean)
        
        # Main loop
        while calls_made < budget:
            remaining = budget - calls_made
            
            # If we can't fit a full population, do greedy steps
            if remaining < lam:
                for _ in range(remaining):
                    y = x_mean + sigma * rng.standard_normal(d)
                    safe_func(y)
                break

            # Sample population
            Y = np.zeros((lam, d))
            for i in range(lam):
                Y[i] = x_mean + sigma * rng.standard_normal(d)

            # Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                F[i] = safe_func(Y[i])

            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Check for convergence/stalling
            if F_sorted[-1] - F_sorted[0] < 1e-10:
                # Restart: reset sigma, keep best_x as new mean
                x_mean = best_x.copy()
                sigma = 2.0
                C = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                gen = 0
                continue

            gen += 1
            # Update mean
            z = np.zeros(d)
            for i in range(mu):
                z += weights[i] * (Y_sorted[i] - x_mean) / sigma
            x_mean = x_mean + sigma * z

            # Update p_sigma
            c_sqrt = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            # Use gen for the exponent to be consistent with generation count
            h_sigma = (np.linalg.norm(z) / np.sqrt(1 - (1 - c_sigma) ** (gen / (c_sigma * (d + 1))))) < (1.4 + 2 / (d + 1)) * np.sqrt((d + 1) / (d - 1))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * c_sqrt * z

            # Update p_c and C
            c_sqrt_c = np.sqrt(c_c * (2 - c_c) * mu_eff)
            h_c = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen / (c_sigma * (d + 1))))) < (1.4 + 2 / (d + 1)) * np.sqrt((d + 1) / (d - 1))
            p_c = (1 - c_c) * p_c + h_c * c_sqrt_c * (z / sigma)

            # Rank-mu update
            H = np.zeros((d, d))
            for i in range(mu):
                y_i = (Y_sorted[i] - x_mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)
            H = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C) + c_mu * H

            # Symmetrize
            C = 0.5 * (H + H.T)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

            # Clip sigma
            if sigma > 10:
                sigma = 10
            if sigma < 1e-8:
                sigma = 1e-8

            # Ensure x_mean is in bounds
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
