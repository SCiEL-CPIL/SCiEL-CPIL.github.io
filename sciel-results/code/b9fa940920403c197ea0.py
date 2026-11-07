"""Optimizer for BBOB f15 (Rastrigin rotated) – Sobol mapping + CMA-ES polish."""
import numpy as np
import time


def _sobol_sequence(n, dim, seed=0):
    """Generate a Sobol quasi-random sequence in [0,1]^dim using scipy."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        samples = sampler.random(n)
        return samples
    except ImportError:
        # Fallback to random if scipy not available
        rng = np.random.default_rng(seed)
        return rng.random((n, dim))


class Optimizer:
    """
    Two-phase optimizer:
      Phase 1: Sobol quasi-random sampling to map the landscape and identify top basins.
      Phase 2: CMA-ES initialized at the best basin to exploit the global minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # --- Phase 1: Sobol quasi-random mapping ---
        # Use a fraction of the budget for global exploration
        sobol_budget = int(budget * 0.35)
        n_sobol = sobol_budget

        # Generate Sobol points in [0,1]^dim and map to [-5,5]^dim
        sobol_samples = _sobol_sequence(n_sobol, dim, seed=123)
        sobol_points = sobol_samples * 10.0 - 5.0

        # Evaluate Sobol points
        evaluated_points = []
        for i in range(n_sobol):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            x = np.clip(sobol_points[i], -5, 5)
            fx = func(x)
            calls_used += 1
            evaluated_points.append((fx, x))

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness and take top 3 basins
        evaluated_points.sort(key=lambda t: t[0])
        top_basins = [x for _, x in evaluated_points[:3]]

        # --- Phase 2: CMA-ES from best basin ---
        remaining_budget = budget - calls_used
        if remaining_budget <= 0 or best_x is None:
            return best_x, best_f

        # CMA-ES parameters
        x_mean = best_x.copy()
        sigma = 1.5  # Large initial step size for Rastrigin
        pop_size = max(4, int(4 + 3 * np.log(dim)))

        # CMA-ES state
        weights = np.ones(pop_size)
        weights[0] = 2.0
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (mu_eff - 2 + 2 / (dim + 2)) / ((dim + 2) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        hs = np.sqrt((1 + c_sigma / c_sigma) * (dim + 1))

        for iteration in range(remaining_budget // pop_size + 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                if calls_used >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((fy, y))

            if len(offspring) < pop_size:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Update mean
            new_mean = sum(weights[i] * offspring[i][1] for i in range(pop_size))

            # Update covariance matrix adaptation
            C = B @ np.diag(D ** 2) @ B.T

            # Update step size control
            y_mean = (new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1 + c_sigma / 2) * hs
            if h_sigma:
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + c_c * (2 - c_c) * C) + c_mu * sum(weights[i] * np.outer(offspring[i][1] - x_mean, offspring[i][1] - x_mean) for i in range(pop_size)) / sigma ** 2
            else:
                p_c = (1 - c_c) * p_c

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / hs - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Eigendecomposition for next iteration
            C = (C + C.T) / 2  # Symmetrize
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            D = np.sqrt(eigenvalues)
            B = eigenvectors

            x_mean = new_mean

        return best_x, best_f
