"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4, integrated restart, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        # CMA-ES parameters
        lam = 4  # Population size
        w = np.ones(lam)
        w[:lam // 2] = 2.0
        w /= w.sum()
        mueff = 1.0 / w.sum()

        # Initial state
        rng = np.random.default_rng(0)
        center = rng.uniform(-5, 5, dim)
        C = np.eye(dim)
        C_eigval, C_eigvec = np.linalg.eigh(C)
        C_eigval = np.clip(C_eigval, 1e-12, None)
        sigma = 2.0

        # Parameters
        cmu = 2.0 / ((dim + 1.7) ** 2 + lam)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = (2 - cmu + 2 * mueff / dim) / ((dim + 1.6) ** 2 + lam)
        cs = (1 + 2 * max(0, mueff - 1) / dim + cc) / (
            dim + 1 + 2 * max(0, mueff - 1) / dim + cc
        )
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cc

        # Track best
        best_x = center.copy()
        best_f = func(center)
        calls_made = 1

        # Restart tracking
        restart_count = 0
        max_restarts = 3

        for generation in range(budget // lam):
            if time.time() > deadline - 5.0:
                break

            # Generate and evaluate population
            pop = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = center + sigma * (C_eigvec @ (np.sqrt(C_eigval) * z))
                y = np.clip(y, -5, 5)
                pop.append(y)

            fvals = np.array([func(x) for x in pop])
            calls_made += lam

            # Sort
            indices = np.argsort(fvals)
            pop_sorted = [pop[i] for i in indices]
            fvals_sorted = fvals[indices]

            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()

            # Update center
            n_top = int(np.floor(mueff))
            w_top = w[:n_top]
            center_new = sum(w_top[i] * pop_sorted[i] for i in range(n_top))

            # Evolution path
            y_vec = (center_new - center) / sigma
            hs = (1 - cs) * hs + np.sqrt(cs * (2 - cs) * mueff) * (
                C_eigvec @ (C_eigval ** (-0.5)) @ (C_eigvec.T @ y_vec)
            )
            hs_norm = np.linalg.norm(hs)

            # Update sigma
            sigma *= np.exp(cs / damps * (hs_norm / np.sqrt(2 * dim) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance
            delta_hs = (1 + cs) * (1 - cc) * hs
            rank_one = np.outer(y_vec, y_vec)
            rank_mu = np.zeros((dim, dim))
            for i in range(n_top):
                yi = (pop_sorted[i] - center) / sigma
                rank_mu += w_top[i] * np.outer(yi, yi)

            C = (
                (1 - c1 - cmu * mueff) * C
                + c1 * (rank_one + delta_hs * hs.T + np.outer(delta_hs, hs))
                + cmu * rank_mu
            )
            C = (C + C.T) / 2.0
            center = center_new

            # Update eigen-decomposition
            C_eigval, C_eigvec = np.linalg.eigh(C)
            C_eigval = np.clip(C_eigval, 1e-12, None)

            # Restart condition
            if sigma < 1e-6 or sigma > 50.0:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                center = rng.uniform(-5, 5, dim)
                C = np.eye(dim)
                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_eigval = np.clip(C_eigval, 1e-12, None)
                sigma = 2.0
                hs = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
