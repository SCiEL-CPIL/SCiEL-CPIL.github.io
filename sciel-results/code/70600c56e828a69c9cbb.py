"""Optimizer for BBOB-style f11 (Discus) using CMA-ES."""
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for the Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(d)))  # population size
        mu = lam // 2  # number of parents

        # Initial point: center of the box
        x = np.zeros(d)
        fx = func(x)

        # Initial step size
        sigma = 2.0

        # Initial covariance matrix
        C = np.eye(d)
        Cinv = np.eye(d)
        Csqrt = np.eye(d)
        Cinv_sqrt = np.eye(d)

        # Evolution paths
        ps = np.zeros(d)
        pc = np.zeros(d)

        # Learning rates
        cs = (lam + 2) / (d + lam + 5)
        cmu = 2 / ((d + 2.) ** 2 + lam)
        c1 = (2 + 2. / ((d + 1.) ** 2)) / ((d + lam) ** 1.6 + 1)
        cmu_eff = cmu
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 + 1) / (d + 1)) - 1) + cs
        dampc = 1 + 2 * max(0, np.sqrt((lam / 2 + 1) / (d + 1) - 1) - 1) + cs

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1. / np.sum(weights ** 2)

        # State variables for the adaptive step size
        chi_n = np.sqrt(d) * (1 - 1. / (4 * d) + 1. / (21 * d ** 2))

        # Track best
        best_x = x.copy()
        best_f = fx

        # Precompute for efficiency
        def matvec(M, v):
            return M @ v

        for gen in range(budget // lam):
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                # Sample from N(x, sigma^2 * C)
                z = np.random.randn(d)
                y = x + sigma * (Csqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                if time.time() > deadline:
                    break

            if len(offspring) < lam:
                # Fill remaining with best if we ran out of time
                while len(offspring) < lam:
                    offspring.append((best_f, best_x.copy()))

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Select parents
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += weights[i] * (offspring[i][1] - x)

            # Update mean
            x_new = x + y_w

            # Update evolution path for covariance
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) / sigma * y_w

            # Update evolution path for step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt @ y_w

            # Update covariance matrix
            delta_h = (1 + cs) * np.sqrt((2 - cs) * mueff) * (np.linalg.norm(ps) ** 2 / d - 1)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                zi = (offspring[i][1] - x) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C = (1 + c1 * delta_h - c1 - cmu_eff) * C + c1 * rank_one + cmu_eff * rank_mu

            # Re-ensure C is positive definite
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except:
                C = np.eye(d)

            # Update inverse and square root
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals_inv = 1.0 / np.maximum(eigvals, 1e-12)
                Cinv = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T
                Csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
            except:
                C = np.eye(d)
                Cinv = np.eye(d)
                Csqrt = np.eye(d)
                Cinv_sqrt = np.eye(d)

            # Update step size
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(cs / damps * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e6)

            x = x_new

            # If converged, restart
            if np.linalg.norm(y_w) / sigma < 1e-10:
                break

        return best_x, best_f
