"""
CMA-ES style optimizer for BBOB f18 (Schaffers F7 ill-conditioned).

Design: Full CMA-ES with population-based search, covariance matrix adaptation,
and step-size control.  In 5D with 500 evaluations, a population of 10 gives
~50 generations, which is enough to find the global minimum of F7.  CMA-ES
handles ill-conditioning and rotated landscapes far better than coordinate
search.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        budget = self.budget
        dim = self.dim
        lam = 10  # population size
        mu = lam // 2  # number of parents

        # Initial point at center
        x_mean = np.zeros(dim)
        f_mean = func(x_mean)
        n_evals = 1

        best_x = x_mean.copy()
        best_f = f_mean

        # Step size: start with a large step to explore the landscape
        sigma = 2.0

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        C_inv = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Learning rates
        c_sigma = (mu + 2) / (dim + 4)
        d_sigma = 1 + 2 * max(0, (dim - 1) / (dim + 3) - 1) + c_sigma
        c_c = (4 + dim / mu) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu - 2 + 1. / mu)) / ((dim + 2) ** 2 + mu))

        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))

        rng = np.random.RandomState(42)

        n_generations = max(1, (budget - n_evals) // lam)

        for gen in range(n_generations):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C @ z)
                # Clip to box
                y = np.clip(y, -5.0, 5.0)
                offspring.append(y)

            # Evaluate offspring
            f_offspring = []
            for y in offspring:
                if n_evals >= budget:
                    break
                fy = func(y)
                n_evals += 1
                f_offspring.append(fy)

            if len(f_offspring) < lam:
                # Pad with worst value
                f_offspring.extend([float('inf')] * (lam - len(f_offspring)))

            # Sort offspring by fitness
            order = np.argsort(f_offspring)
            offspring = [offspring[i] for i in order]
            f_offspring = [f_offspring[i] for i in order]

            # Update best
            for y, fy in zip(offspring, f_offspring):
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Update mean
            x_new_mean = sum(weights[i] * offspring[i] for i in range(mu))

            # Update step size (CSA)
            y_mean = (x_new_mean - x_mean) / sigma if sigma > 0 else np.zeros(dim)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ y_mean)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            delta_heav = (1 + (1 - c_c) * (1 + c_sigma) * np.linalg.norm(p_sigma) ** 2 / chi_n - 1) \
                         if np.linalg.norm(p_sigma) > 0 else 0.0
            rank_one = np.outer(p_c, p_c)
            C = (1 + c_1 * (delta_heav - 1)) * C + c_1 * rank_one

            # Rank-mu update
            diff = [offspring[i] - x_mean for i in range(mu)]
            rank_mu = sum(weights[i] * np.outer(diff[i], diff[i]) for i in range(mu))
            C = (1 - c_1 - c_mu) * C + c_mu * rank_mu

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Invert C for next iteration
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)

            x_mean = x_new_mean

        return best_x.tolist(), best_f
