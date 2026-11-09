"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters tuned for 500 evals, 5D, Weierstrass f16
        # Using smaller population for more generations and better exploration
        lambda_ = 8  # population size: ~62 generations
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
        cs = (dim + 1.7) / (dim + 9)
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - cc)
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        # Initialize with wider initial spread for Weierstrass multi-scale landscape
        x_mean = np.zeros(dim)
        sigma = 2.0  # Wider initial step to explore the landscape
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        evals_used = 0
        rng = np.random.default_rng()
        generation = 0

        def evaluate(x):
            nonlocal evals_used
            fx = func(x)
            evals_used += 1
            return fx

        # Initial evaluation
        fx = evaluate(x_mean)
        best_x = x_mean.copy()
        best_f = fx

        while evals_used + lambda_ <= budget:
            generation += 1
            # Generate population
            x_pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                x_pop[i] = x_mean + sigma * (B @ (D * z))

            # Evaluate
            f_pop = np.array([evaluate(x_pop[i]) for i in range(lambda_)])

            # Update best
            idx_min = np.argmin(f_pop)
            if f_pop[idx_min] < best_f:
                best_f = f_pop[idx_min]
                best_x = x_pop[idx_min].copy()

            # Sort by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]

            # Weighted recombination
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update path evolution strategies
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (x_new - x_mean)))
            # Correct hsig: use cumulative sum of (1-cs) over generations
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * generation))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new - x_mean)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None] * np.outer(x_sorted[:mu] - x_mean, x_sorted[:mu] - x_mean), axis=0)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Ensure symmetry and positive semi-definite (single eigendecomp)
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))

            # Update mean
            x_mean = x_new

            # Check for convergence
            if sigma < 1e-12:
                break

        # Use remaining budget for local refinement around best_x
        remaining = budget - evals_used
        if remaining > 0:
            # Simple local search with decreasing step size
            local_sigma = sigma * 0.5 if sigma > 0 else 0.1
            for _ in range(remaining):
                z = rng.standard_normal(dim)
                x_trial = best_x + local_sigma * z
                f_trial = evaluate(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    local_sigma *= 0.95
                else:
                    local_sigma *= 0.98

        return best_x, best_f
# EVOLVE-BLOCK-END
