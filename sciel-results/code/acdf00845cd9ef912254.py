"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill

        best_x = None
        best_f = np.inf

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        dim = self.dim
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # standard CMA-ES population size
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size control
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (2 - cc)
        cmu = (2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)) * (
            min(1, c1) - 1
        )
        cmu = max(0, min(cmu, 0.5))
        # Damping
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # Sigma
        sigma0 = 2.0  # initial step size
        # Boundary handling: we'll clip

        def cma_es(x0, sigma, budget_remaining, rng_local):
            nonlocal best_x, best_f
            # State
            x_mean = x0.copy()
            sigma = sigma
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            # Initial evaluations
            f_mean = func(x_mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()
            calls_used = 1

            # Track calls
            while calls_used < budget_remaining:
                if time.time() > deadline:
                    break

                # Generate population
                y = rng_local.standard_normal((lambda_, dim))
                z = y @ (B * D)  # B @ D @ y.T shape
                # Actually B is rotation, D is diagonal scaling
                # z_i = B @ (D * y_i)
                z = (y * D) @ B.T
                x_pop = x_mean + sigma * z
                # Clip to bounds
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = np.array([func(x_pop[i]) for i in range(lambda_)])
                calls_used += lambda_

                # Update best
                idx_best = np.argmin(f_pop)
                if f_pop[idx_best] < best_f:
                    best_f = f_pop[idx_best]
                    best_x = x_pop[idx_best].copy()

                # Sort by fitness
                idx_sorted = np.argsort(f_pop)
                y_sorted = y[idx_sorted]
                f_sorted = f_pop[idx_sorted]

                # Weighted mean of best mu
                x_new = np.sum(weights[:, None] * x_pop[idx_sorted[:mu]], axis=0)

                # Update covariance matrix adaptation
                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D @ B.T @ (
                    (x_new - x_mean) / sigma
                )
                # hsig condition
                hsig = (
                    np.linalg.norm(ps)
                    / np.sqrt(1 - (1 - cs) ** (2 * (calls_used + 1) / (dim + mueff)))
                    < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1)
                )
                # Update C
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (
                    (x_new - x_mean) / sigma
                )
                # Rank-one and rank-mu updates
                delta = (1 + hsig * cs * (1 - cs)) * (
                    np.outer(pc, pc) - np.eye(dim)
                )
                C_update = (
                    c1 * delta
                    + cmu * np.sum(
                        weights[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]),
                        axis=0,
                    )
                    - (c1 + cmu) * np.eye(dim)
                )
                # Update B and D via eigen decomposition (approximate for small dim)
                # For dim=5, we can do full eigendecomp
                eigvals, eigvecs = np.linalg.eigh(B @ (D ** 2) @ B.T + C_update)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                # Update sigma
                sigma *= np.exp(
                    cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1)
                )
                # Update mean
                x_mean = x_new
                f_mean = f_sorted[0]

                # Boundary handling: if many points are at boundary, reduce sigma
                # Check if mean is at boundary
                at_boundary = np.sum(np.abs(x_mean + 5) < 1e-8) + np.sum(
                    np.abs(x_mean - 5) < 1e-8
                )
                if at_boundary > dim // 2:
                    sigma *= 0.5

            return x_mean, f_mean

        # Multiple restarts
        n_restarts = 5
        per_restart_budget = self.budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if best_x is not None:
                # Restart from best with perturbation
                x0 = best_x + rng.normal(0, 0.5, dim)
                x0 = np.clip(x0, -5, 5)
                sigma_start = 0.5
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma_start = 2.0

            cma_es(x0, sigma_start, per_restart_budget, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
