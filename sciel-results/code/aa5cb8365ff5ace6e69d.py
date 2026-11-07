"""CMA-ES optimizer for BBOB f12 (Bent Cigar) with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance for anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1,
                   2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-4, 4, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        inv_sigma = 1.0 / sigma
        C_sqrt_inv = np.eye(dim)  # C^{-1/2}
        C_sqrt = np.eye(dim)      # C^{1/2}

        best_x = x_mean.copy()
        best_f = func(best_x)

        calls = 1

        for gen in range(budget // lambda_ + 10):
            if time.time() > deadline - 5:
                break
            if calls >= budget:
                break

            # Generate population
            population = np.empty((lambda_, dim))
            fitness = np.empty(lambda_)

            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline - 1:
                    break
                z = rng.standard_normal(dim)
                x = x_mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                population[i] = x
                fitness[i] = func(x)
                calls += 1

            # Sort by fitness
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = population[0].copy()

            # Weighted mean of selected individuals
            x_new = np.sum(weights[:, None] * population[:mu], axis=0)

            # Update step size control vector
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                 (inv_sigma * (C_sqrt_inv @ (x_new - x_mean)))

            # Update evolution path for covariance
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < \
                   (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * \
                 (inv_sigma * (x_new - x_mean))

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 - c_1 - c_mu + c_1 * delta_hsig) * C
            C += c_1 * (np.outer(pc, pc) - delta_hsig * C)
            C += c_mu * np.sum(weights[:, None, None] *
                               np.einsum('i,ij,ik->ijk',
                                         (population[:mu] - x_mean) / sigma,
                                         (population[:mu] - x_mean) / sigma,
                                         (population[:mu] - x_mean) / sigma),
                               axis=0) / mu_eff

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e4)

            # Update mean
            x_mean = x_new
            inv_sigma = 1.0 / sigma

            # Recompute C^{1/2} and C^{-1/2} periodically
            if gen % 10 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    C_sqrt_inv = np.eye(dim)

            # If we've used most of our budget, do a final local refinement
            if calls >= budget - 20 and time.time() < deadline - 2:
                # Nelder-Mead style polish
                nm_points = [best_x.copy()]
                for j in range(dim):
                    p = best_x.copy()
                    p[j] += 0.1
                    nm_points.append(p)
                    p = best_x.copy()
                    p[j] -= 0.1
                    nm_points.append(p)

                for _ in range(10):
                    if calls >= budget:
                        break
                    if time.time() > deadline - 1:
                        break
                    # Simple coordinate descent polish
                    for j in range(dim):
                        if calls >= budget:
                            break
                        if time.time() > deadline - 1:
                            break
                        for step in [0.05, 0.02, 0.01, 0.005, 0.001]:
                            for direction in [1, -1]:
                                y = best_x.copy()
                                y[j] += direction * step
                                y = np.clip(y, -5, 5)
                                fy = func(y)
                                calls += 1
                                if fy < best_f:
                                    best_x = y
                                    best_f = fy

                break

        return best_x, best_f
# EVOLVE-BLOCK-END
