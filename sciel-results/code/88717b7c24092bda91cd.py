"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population size for f14 optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        remaining = self.budget
        
        # Use smaller population for more generations (exploitation)
        lam = 4
        mu = max(1, lam // 2)

        # Weights
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 1) - np.log(i + 1)
        weights /= weights.sum()
        mueff = int(np.sum(weights)**2 / np.sum(weights**2))

        # Strategy parameters (standard CMA-ES)
        c1 = (2 * (2 - mueff / (n + 2))**2) / ((n + 2)**2 + 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + 2))
        cs = (mueff + 2) / (n + mueff + 5)
        c1 += 2 * (1 - c1 - cmu) * (mueff / (n + 2) - 1)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Initial values - start at origin with large sigma to cover box
        xmean = np.zeros(n)
        sigma = 5.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        generation = 0

        # Initial evaluation at origin
        fx = func(xmean)
        best_x = xmean.copy()
        best_f = fx
        remaining -= 1

        # Main CMA-ES loop
        while remaining >= lam:
            generation += 1
            # Generate population
            z = rng.standard_normal((lam, n))
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
            except np.linalg.LinAlgError:
                L = np.eye(n)

            x_pop = np.array([xmean + sigma * (z[i] @ L) for i in range(lam)])
            x_pop = np.clip(x_pop, -5, 5)

            # Evaluate
            f_pop = np.array([func(x_pop[i]) for i in range(lam)])
            remaining -= lam

            # Update best
            min_idx = np.argmin(f_pop)
            if f_pop[min_idx] < best_f:
                best_f = f_pop[min_idx]
                best_x = x_pop[min_idx].copy()

            # Sort by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]

            # Update mean using weighted recombination
            xnew = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            dx = xnew - xmean

            # Update evolution paths
            norm_L = np.linalg.norm(L)
            if norm_L > 1e-10:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / norm_L * (L @ (dx / sigma))
            else:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (dx / sigma)
            
            # Correct hsig calculation using generation counter
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * generation)) / chiN) < (1.4 + 2 / (n + 1))
            
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * dx / sigma

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c1 * (2 - c1)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - C)
            C += cmu / (sigma**2) * np.sum(weights[:, None, None] * np.outer(x_sorted[:mu] - xmean, x_sorted[:mu] - xmean), axis=0)

            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 5.0)

            # Update mean
            xmean = xnew

        # Use remaining budget for local refinement around best_x
        # Use a simple (1+1)-ES style local search
        local_sigma = 0.5
        while remaining > 0:
            step = rng.standard_normal(n) * local_sigma
            y = np.clip(best_x + step, -5, 5)
            fy = func(y)
            remaining -= 1
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                local_sigma *= 1.2
            else:
                local_sigma *= 0.7
            local_sigma = max(1e-6, min(local_sigma, 2.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
