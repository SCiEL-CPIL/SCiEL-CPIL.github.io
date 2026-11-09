"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic global-to-local hierarchical search: coarse grid then CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse grid scan using a 3^5 grid (243 points) over [-5, 5]^5
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            grid_points.append(np.array([i * 5 - 5, j * 5 - 5, k * 5 - 5,
                                                         l * 5 - 5, m * 5 - 5], dtype=float))

        # Evaluate all grid points
        best_x = None
        best_f = np.inf
        for point in grid_points:
            fx = func(point)
            if fx < best_f:
                best_f = fx
                best_x = point.copy()

        calls_used = len(grid_points)

        # Phase 2: Localized CMA-ES around the best grid point
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        x = best_x.copy()
        sigma = 1.5
        C = np.eye(dim)
        C_cholesky = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        cs = 1 / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + 10)
        cmu = min(1 - c1, 2 * ((dim - 2) + 2 / (dim + 1)) / ((dim + 2) ** 2 + 15))
        cmu = min(cmu, 1 - c1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        lambda_ = min(4 * dim, max(4, remaining // 2))
        mu_eff = max(1, int(np.floor(lambda_ / 2)))
        weights = np.ones(mu_eff)
        weights[0] = 2
        weights /= weights.sum()
        mu_eff_w = (weights.sum()) ** 2 / np.sum(weights ** 2)

        rng = np.random.default_rng(42)

        gen = 0
        while calls_used < budget:
            # Generate offspring
            offspring = []
            offspring_f = []
            for _ in range(lambda_):
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C_cholesky @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_f.append(fy)
                calls_used += 1

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)
            indices = np.argsort(offspring_f)
            sorted_offspring = offspring[indices]
            sorted_f = offspring_f[indices]

            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            new_mean = np.sum(weights[:, None] * sorted_offspring[:mu_eff], axis=0)

            # Update paths
            diff = (new_mean - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff_w) * diff
            norm_ps = np.linalg.norm(ps)
            hsig = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff_w) * diff

            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            outer_pc = np.outer(pc, pc)
            outer_diffs = np.sum(weights[:, None, None] *
                                 np.einsum('i,ij,ik->ijk', np.ones(mu_eff),
                                           sorted_offspring[:mu_eff] - x,
                                           sorted_offspring[:mu_eff] - x), axis=0)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * outer_pc + cmu * outer_diffs

            # Ensure positive semi-definite and update Cholesky
            try:
                C_cholesky = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_cholesky = np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / (dim + 4) * (norm_ps / chi_n - 1))
            sigma = min(max(sigma, 1e-10), 10.0)

            x = new_mean
            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
