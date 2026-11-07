"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with path-based step-size adaptation for bent cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin

        # Parameters
        lambda_ = 4 * int(np.ceil(np.sqrt(dim)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(1 / np.sum(weights ** 2))
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialization: centered near origin since minimizer is in [-4,4]^5
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0

        # Initial evaluations to get a starting point
        # Evaluate a small set of initial points to find a good starting location
        n_init = min(10, budget)
        best_x = None
        best_f = np.inf
        for i in range(n_init):
            if time.time() > deadline:
                break
            if i == 0:
                x_candidate = np.zeros(dim)
            else:
                x_candidate = rng.uniform(-4, 4, dim)
            fx = func(x_candidate)
            if fx < best_f:
                best_f = fx
                best_x = x_candidate.copy()

        x_mean = best_x.copy()

        # CMA-ES state
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        calls_made = n_init

        for gen in range(budget - calls_made):
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lambda_, dim))
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                x_off = x_mean + sigma * (B @ (D * z))
                x_off = np.clip(x_off, -5, 5)
                offspring[i] = x_off
                fx = func(x_off)
                calls_made += 1
                offspring[i, dim] = fx  # store function value

            # Sort by function value
            f_values = offspring[:, dim]
            order = np.argsort(f_values)
            offspring = offspring[order]
            f_values = f_values[order]

            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = offspring[0, :dim].copy()

            # Update mean
            x_new = np.sum(weights[:, None] * offspring[:mu, :dim], axis=0)
            x_mean = x_new

            # Update covariance matrix adaptation
            y_vec = (x_new - x_mean) / sigma if sigma > 1e-15 else np.zeros(dim)
            pc = (1 - c_c) * pc + c_c * y_vec
            norm_pc = np.linalg.norm(pc)
            h_sig = (norm_pc / np.sqrt(1 - (1 - c_c) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n

            ps = (1 - c_sigma) * ps + (h_sig / chi_n) * y_vec
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 20.0)

            # Covariance matrix update
            delta_h = (1 - h_sig) * c_c * (2 - c_c)
            outer_pc = np.outer(pc, pc)
            D2 = D ** 2
            C = (B @ np.diag(D2) @ B.T)
            C = ((1 + c_1 + c_mu - c_1 - c_1) * C
                 + c_1 * (dim + delta_h - 1) * C
                 + c_1 * (outer_pc - C)  # approximation
                 + c_mu * np.sum(weights[:, None, None] * np.outer(offspring[:mu, :dim] - x_mean, offspring[:mu, :dim] - x_mean), axis=0) / sigma ** 2)

            # Eigen decomposition for B and D
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
