"""Optimizer for BBOB f19 (Griewank-Rosenbrock) with CMA-ES + time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        # Population size
        lam = max(4, int(np.ceil(4 + 3 * np.log(d))))
        mu = lam // 2

        # Initial point and step size
        rng = np.random.default_rng(42)
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 2.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        c_sigma = 1.0 / (lam + 3)
        c_c = (2 * (mu + 1) - 1) / ((d + 2) ** 2 + lam)
        mu_w = np.arange(mu, 0, -1)
        mu_w = mu_w.astype(float)
        mu_w = mu_w / mu_w.sum()
        c_mu = np.sum(mu_w[:min(2, mu)]) / np.sum(mu_w)

        calls = 1

        for gen in range(1, budget + 1):
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            population = []
            for i in range(lam):
                z = rng.standard_normal(d)
                # CMA-ES sampling: x = x + sigma * C^(1/2) * z
                # Use Cholesky for C^(1/2)
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    # Fallback to diagonal
                    y = np.sqrt(np.diag(C)) * z
                y_i = x + sigma * y
                population.append(y_i)

            # Evaluate
            fitness = []
            for y_i in population:
                if time.time() > deadline - 2.0:
                    # Fill remaining with worst
                    fitness.append(np.inf)
                else:
                    f_i = func(y_i)
                    calls += 1
                    fitness.append(f_i)
                    if f_i < best_f:
                        best_f = f_i
                        best_x = y_i.copy()

            fitness = np.array(fitness)

            # Sort by fitness
            order = np.argsort(fitness)
            selected = [population[order[i]] for i in range(mu)]
            selected_fitness = [fitness[order[i]] for i in range(mu)]

            # Weighted mean
            m = np.zeros(d)
            for i, (w, y_i) in enumerate(zip(mu_w, selected)):
                m += w * y_i

            # Update sigma
            diff = m - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update C and p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
            C = (1 - c_c * (2 - c_c)) * C + c_c * (p_c[:, None] * p_c[None, :] - C)

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Update mean
            x = m

            # Check convergence
            if sigma < 1e-12 or not np.isfinite(best_f):
                break

        return best_x, best_f

    @staticmethod
    def _mu_eff(mu, lam):
        return mu * (lam - mu) / (lam - 1) if lam > 1 else mu


# Fix: mu_eff should be computed inline
class Optimizer:
    """CMA-ES style optimizer with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        d = self.dim
        budget = self.budget

        lam = max(4, int(np.ceil(4 + 3 * np.log(d))))
        mu = lam // 2
        mu_eff = mu * (lam - mu) / (lam - 1) if lam > 1 else mu

        rng = np.random.default_rng(42)
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 2.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        c_sigma = 1.0 / (lam + 3)
        c_c = (2 * (mu + 1) - 1) / ((d + 2) ** 2 + lam)
        mu_w = np.arange(mu, 0, -1, dtype=float)
        mu_w = mu_w / mu_w.sum()

        calls = 1

        for gen in range(1, budget + 1):
            if time.time() > deadline - 5.0:
                break

            population = []
            for i in range(lam):
                z = rng.standard_normal(d)
                try:
                    L = np.linalg.cholesky(C)
                    y = L @ z
                except np.linalg.LinAlgError:
                    y = np.sqrt(np.diag(C)) * z
                y_i = x + sigma * y
                population.append(y_i)

            fitness = []
            for y_i in population:
                if time.time() > deadline - 2.0:
                    fitness.append(np.inf)
                else:
                    f_i = func(y_i)
                    calls += 1
                    fitness.append(f_i)
                    if f_i < best_f:
                        best_f = f_i
                        best_x = y_i.copy()

            fitness = np.array(fitness)
            order = np.argsort(fitness)
            selected = [population[order[i]] for i in range(mu)]

            m = np.zeros(d)
            for i, (w, y_i) in enumerate(zip(mu_w, selected)):
                m += w * y_i

            diff = m - x
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
            C = (1 - c_c * (2 - c_c)) * C + c_c * (p_c[:, None] * p_c[None, :] - C)
            C = 0.5 * (C + C.T)

            x = m

            if sigma < 1e-12 or not np.isfinite(best_f):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
