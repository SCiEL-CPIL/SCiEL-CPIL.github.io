"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with LHS initialization and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube Sampling for global exploration
        # Use ~80 evaluations for LHS
        n_lhs = min(80, budget // 6)
        if n_lhs < 5:
            n_lhs = 5

        # Generate LHS points in [-5, 5]^dim
        rng = np.random.default_rng(42)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_points[:, d] = -5.0 + 10.0 * (perm + rng.random(n_lhs)) / n_lhs

        best_x = None
        best_f = np.inf
        eval_count = 0

        for i in range(n_lhs):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from the best LHS point
        remaining = budget - eval_count
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # CMA-ES parameters
        x = best_x.copy()
        f_best = best_f
        sigma = 1.0  # initial step size
        lambda_ = max(4 + int(3 * np.log(dim)), dim + 2)  # population size
        mu = lambda_ // 2  # number of selected individuals

        # Covariance matrix and its Cholesky decomposition
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        # Eigenvalue decomposition for inverse square root
        eigvals, eigvecs = np.linalg.eigh(C)
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T

        # Adaptation parameters
        cc = (4 + lambda_ / dim) / ((dim + 1.7) ** 2 + lambda_ / 2)
        cs = (lambda_ / 2 + 1) / (dim + 2 + lambda_ / 2)
        cmu = min(1 - 2 * cc, 2 * ((mu + 1.7) / (dim + 2) - 1) / ((dim + 2) ** 2 + lambda_ / 2))
        damps = 1 + 2 * max(0, np.sqrt((lambda_ / mu - 1) * (cs - 1) / dim) - 1) + cs
        dampc = (1 + 2 * max(0, np.sqrt((lambda_ / mu - 1) * (cc - 1) / dim) - 1)) / \
                (dim + 1 + 2 * max(0, np.sqrt((lambda_ / mu - 1) * (cc - 1) / dim)))

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Initial population
        pop = np.zeros((lambda_, dim))
        fit = np.zeros(lambda_)

        for it in range(remaining):
            if time.time() > deadline:
                break

            # Generate population
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                pop[i] = x + sigma * (C_inv_sqrt @ z)
                pop[i] = np.clip(pop[i], -5, 5)
                fit[i] = func(pop[i])
                eval_count += 1

            if eval_count >= budget:
                break

            # Sort by fitness
            idx = np.argsort(fit)
            # Select top mu
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = w.sum() ** 2

            # Update mean
            x_old = x.copy()
            x_new = sum(w[i] * pop[idx[i]] for i in range(mu))

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ ((x_new - x_old) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1)))) / \
                   np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))
            if hsig:
                pc = (1 - cc) * pc + cc / dampc * ((x_new - x_old) / sigma)
            else:
                pc = (1 - cc) * pc

            # Update covariance matrix
            delta_hsig = (1 if hsig else 0) - (1 - cc)
            rank_mu_matrix = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (pop[idx[i]] - x_old) / sigma
                rank_mu_matrix += w[i] * np.outer(y_i, y_i)

            C = (1 + cmu * mueff / dampc * (np.outer(pc, pc) - C) + cmu * rank_mu_matrix) * \
                (1 - cmu * mueff / dampc)
            # Ensure positive semi-definite
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.maximum(eigvals_C, 1e-20)
            C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) - 1))
            sigma = max(sigma, 1e-10)

            # Update mean
            x = x_new

            # Update Cholesky-like inverse square root
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Track best
            if fit[idx[0]] < f_best:
                f_best = fit[idx[0]]
                best_x = pop[idx[0]].copy()

        # Phase 3: Final polish with (1+1)-ES if budget remains
        remaining2 = budget - eval_count
        if remaining2 > 10 and best_x is not None:
            x = best_x.copy()
            fx = f_best
            sigma = 0.1
            for _ in range(remaining2):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                eval_count += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < f_best:
                        f_best = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, f_best
# EVOLVE-BLOCK-END
