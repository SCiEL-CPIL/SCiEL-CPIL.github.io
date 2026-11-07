"""Optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Initial population: spread out to cover the box
        rng = np.random.default_rng(42)

        # Start near center with moderate spread
        x0 = np.zeros(dim)
        sigma = 3.0

        # CMA-ES parameters
        pop_size = dim + 1 + int(2 * np.log(dim))  # ~10 for dim=5
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 + 2 / (dim + 1) - 1) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.3) / (dim + 5)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        mean = x0.copy()
        C = np.eye(dim)
        Cinv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = 3.0

        # Population
        best_x = mean.copy()
        best_f = func(best_x)
        call_count = 1

        # Track best for early stopping
        no_improve_count = 0
        patience = 50  # generations without improvement

        for gen in range(budget // pop_size + 5):
            if time.time() > deadline:
                break

            # Generate population
            W = np.zeros((pop_size, dim))
            F = np.zeros(pop_size)
            for i in range(pop_size):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                x = mean + sigma * (C @ z)
                x = np.clip(x, -5, 5)
                F[i] = func(x)
                W[i] = x
                call_count += 1
                if call_count >= budget:
                    break

            if call_count >= budget:
                break

            # Update best
            for i in range(pop_size):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = W[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            W = W[order]
            F = F[order]

            # Weights for recombination
            w = np.log(pop_size / 2) - np.log(np.arange(1, pop_size + 1))
            w /= w.sum()
            mueff = int(np.sum(w) ** 2 / np.sum(w ** 2))

            # Update mean
            old_mean = mean.copy()
            mean = np.sum(w[:, None] * W, axis=0)

            # Update path covariance
            pc = (1 - cs) * pc + cs * (mean - old_mean) / sigma
            pc_norm = np.linalg.norm(pc)

            # Update path evolution strategy
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) / pc_norm * (mean - old_mean) / sigma

            # Update covariance
            delta_hs = (1 + (1 - 2 * (pc_norm ** 2 < (1.4 + 2 / (dim + 1)) * chi * chi)) )
            C = ((1 - mu - c1) * C +
                 c1 * (np.outer(pc, pc) + (pc_norm ** 2 > (1.4 + 2 / (dim + 1)) * chi * chi) * (chi ** 2 - 1) * C) +
                 cmu * np.sum(w[:mueff, None, None] * np.outer(W[:mueff] - old_mean, W[:mueff] - old_mean), axis=0))

            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                Cinv = np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Check for convergence
            if np.linalg.norm(mean - old_mean) < 1e-10 * sigma:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count > patience:
                break

        # Final local polish with Nelder-Mead
        if time.time() < deadline - 10:
            from scipy.optimize import minimize
            try:
                res = minimize(func, best_x, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
