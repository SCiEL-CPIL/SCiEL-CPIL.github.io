"""Model-and-optimize optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for BBOB-style f6 (Attractive sector).
    - Population-based sampling with covariance matrix adaptation
    - Handles rotated/shifted landscapes through learned correlations
    - Explicit deadline guard for safety
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        pop_size = 10  # lambda in CMA-ES notation
        mu = pop_size // 2  # number of weights used for recombination
        weights = np.log(np.arange(mu, 0, -1) + 1)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initial state
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Precompute B and D for sqrt(C) decomposition
        def update_sqrt_C(C):
            D, V = np.linalg.eigh(C)
            D = np.clip(D, 1e-12, None)
            B = V @ np.diag(np.sqrt(D))
            return B

        B = update_sqrt_C(C)

        while calls_made < budget:
            if time.time() > deadline - 10.0:
                break

            # Generate population
            offspring = []
            offspring_f = []

            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_f.append(fy)

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in idx]
            sorted_f = [offspring_f[i] for i in idx]

            # Update mean
            new_x = np.sum([w * y for w, y in zip(weights, sorted_offspring[:mu])], axis=0)
            new_x = np.clip(new_x, -5, 5)

            # Update sigma (CSA)
            y_vec = (new_x - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2)) / chi_n < 1.4 + 2 / (dim + 1)
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update evolution paths
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
            C += c_mu * np.sum([w * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) / sigma**2
                                 for i, w in enumerate(weights)], axis=0)

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            try:
                B = update_sqrt_C(C)
            except:
                C = np.eye(dim)
                B = np.eye(dim)

            x = new_x

        return best_x, best_f
# EVOLVE-BLOCK-END
