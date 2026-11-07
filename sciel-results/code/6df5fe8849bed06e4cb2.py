"""Optimized CMA-ES with adaptive population and local refinement for Bent Cigar."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive population size and local search for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Adaptive population: start smaller, grow as we find good regions
        # For Bent Cigar, smaller populations can be effective early on
        lam = 6  # Start with smaller population for faster exploration
        mu = max(2, lam // 2)

        # Precompute weights
        def compute_weights(mu):
            w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            return w

        weights = compute_weights(mu)
        mu_eff = 1 / np.sum(weights ** 2)

        # CMA-ES parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_norm = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (16 * dim ** 2))

        # Track progress for adaptive behavior
        stagnation_count = 0
        last_best_f = best_f

        while calls < budget:
            # Check if we can do a full generation
            if budget - calls < lam:
                # Use remaining budget for local refinement around best point
                remaining = budget - calls
                # Local search with shrinking step size
                local_sigma = max(sigma * 0.1, 1e-6)
                for _ in range(remaining):
                    y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        local_sigma *= 0.8  # Shrink further if we found improvement
                break

            # Generate population
            pop = np.zeros((lam, dim))
            L = np.linalg.cholesky(C)
            for i in range(lam):
                z = rng.standard_normal(dim)
                pop[i] = x + sigma * L.T @ z
                pop[i] = np.clip(pop[i], -5, 5)

            # Evaluate population
            fpop = np.zeros(lam)
            for i in range(lam):
                fpop[i] = func(pop[i])
                calls += 1

            # Update best
            min_idx = np.argmin(fpop)
            if fpop[min_idx] < best_f:
                best_f = fpop[min_idx]
                best_x = pop[min_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # If stagnating, do a local restart around best point
            if stagnation_count > 3:
                # Reset mean to best point and reduce sigma
                x = best_x.copy()
                sigma *= 0.7
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                stagnation_count = 0
                continue

            # Sort population by fitness
            order = np.argsort(fpop)
            pop = pop[order]
            fpop = fpop[order]

            # Compute weighted mean
            x_w = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update step size (sigma)
            L_inv = np.linalg.inv(L)
            C_inv_sqrt = L_inv.T @ L_inv

            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * C_inv_sqrt @ (x_w - x) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_norm - 1))

            # Update pc
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * C_inv_sqrt @ (x_w - x) / sigma

            # Update C
            delta_sigma = (c_c - 2 + c_c ** 2) * sigma ** 2 * (np.linalg.norm(pc) ** 2 / dim - 1)
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_sigma * np.eye(dim))
            for i in range(mu):
                diff = pop[i] - x
                C += c_mu * weights[i] * np.outer(diff, diff)

            # Update mean
            x = x_w

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-8 * np.eye(dim)

            # Adapt population size: if converging well, keep it; if struggling, reduce
            if sigma < 0.1 and best_f < last_best_f:
                # Converging well, can increase population for exploitation
                if lam < 12:
                    lam += 1
                    mu = max(2, lam // 2)
                    weights = compute_weights(mu)
                    mu_eff = 1 / np.sum(weights ** 2)
                    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
                    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                    c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
                    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            elif stagnation_count > 2 and lam > 4:
                # Struggling, reduce population for faster exploration
                lam = max(4, lam - 1)
                mu = max(2, lam // 2)
                weights = compute_weights(mu)
                mu_eff = 1 / np.sum(weights ** 2)
                c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
                d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
                c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
                c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            last_best_f = best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
