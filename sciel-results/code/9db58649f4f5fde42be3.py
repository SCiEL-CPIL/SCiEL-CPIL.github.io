"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size and restarts for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        best_x = None
        best_f = np.inf

        # We'll use multiple restarts to handle potential local minima
        # Each restart uses a fraction of the budget
        num_restarts = 3
        budget_per_restart = self.budget // num_restarts

        rng = np.random.default_rng(42)

        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            # Initialize population
            if restart == 0:
                # First restart: center at origin
                x0 = np.zeros(self.dim)
            else:
                # Subsequent restarts: random positions
                x0 = rng.uniform(-4, 4, self.dim)

            # CMA-ES parameters
            lam = 4 + int(3 * np.log(self.dim))  # population size
            mu = lam // 2  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights @ weights)

            # Step size
            sigma = 2.0

            # Covariance matrix
            C = np.eye(self.dim)
            # Evolution paths
            ps = np.zeros(self.dim)
            pc = np.zeros(self.dim)

            # Update rates
            cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
            c1 = (2 - 1 / (mu_eff + 1) + 2 / (self.dim - 1)) / ((self.dim + 2) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1) - 1)) + cs

            # Initial population
            pop = x0 + sigma * rng.multivariate_normal(np.zeros(self.dim), C, size=lam)
            pop = np.clip(pop, -5, 5)
            fvals = np.array([func(x) for x in pop])

            # Track best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = pop[idx_best].copy()

            # Main loop
            for iteration in range(budget_per_restart - lam):
                if time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                y = pop[order]
                fvals_sorted = fvals[order]

                # Mean of parents
                m = (weights[:, None] * y[:mu]).sum(axis=0)

                # Update evolution paths
                # Normalize C for path updates
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (m - x0) / sigma
                pc = (1 - cs) * pc + (1 - cmu) * (1 - cs) * np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (m - x0) / sigma

                # Update sigma (1/5 rule variant)
                norm_ps = np.linalg.norm(ps)
                chitab = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
                sigma *= np.exp(cs / damps * (norm_ps / chitab - 1))

                # Update covariance
                v = (m - x0) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (1 - (1 - cs) ** 2) * C) \
                    + cmu * sum(w * np.outer(v_i, v_i) for w, v_i in zip(weights, y[:mu] - x0))

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                try:
                    # Eigenvalue correction
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = (eigvecs * eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)

                # Update center
                x0 = m

                # Generate new population
                pop = x0 + sigma * rng.multivariate_normal(np.zeros(self.dim), C, size=lam)
                pop = np.clip(pop, -5, 5)
                fvals = np.array([func(x) for x in pop])

                # Track best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < best_f:
                    best_f = fvals[idx_best]
                    best_x = pop[idx_best].copy()

                # Check for convergence
                if np.all(np.abs(pop - x0) < 1e-8):
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
