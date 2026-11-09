"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        x_best = None
        f_best = np.inf
        calls_made = 0

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of parents
        # Weighting
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size adaptation
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 2) / (dim + mueff)) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def run_cmaes(x0, sigma0, max_evals, rng_local):
            nonlocal x_best, f_best, calls_made
            if max_evals < lambda_ + 1:
                # Not enough for full CMA-ES, use simple (1+1) or (lambda+0)
                x = x0.copy()
                if calls_made >= budget:
                    return
                fx = func(x)
                calls_made += 1
                if fx < f_best:
                    f_best, x_best = fx, x.copy()
                sigma = sigma0
                remaining = max_evals - 1
                for i in range(remaining):
                    if time.time() > deadline or calls_made >= budget:
                        return
                    y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < f_best:
                        f_best, x_best = fy, y.copy()
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                return

            # Full CMA-ES
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            eval_count = 0
            gen = 0

            while eval_count < max_evals and time.time() < deadline and calls_made < budget:
                # Generate population
                B = rng_local.standard_normal((lambda_, dim))
                # Transform B using Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular
                    L = np.eye(dim)
                    C = np.eye(dim)
                Y = (B @ L.T).T  # lambda x dim
                # Actually, we want y = x_mean + sigma * B @ L where L is such that L @ L.T = C
                # Standard: y_i = x_mean + sigma * sum_j B[i,j] * L[j,:]
                # Let's use: Y = x_mean + sigma * (B @ L)
                Y = x_mean[None, :] + sigma * (B @ L)
                Y = np.clip(Y, -5, 5)

                # Evaluate
                F = []
                for y in Y:
                    if calls_made >= budget:
                        break
                    F.append(func(y))
                    calls_made += 1
                
                if len(F) < lambda_:
                    # If we ran out of budget mid-population, pad with inf to keep shape, but logic might break.
                    # Better to just stop if we can't complete the population?
                    # Or just use what we have? CMA-ES needs full population.
                    # If we break here, we should probably return.
                    return
                
                F = np.array(F)
                eval_count += lambda_

                # Update best
                idx_min = np.argmin(F)
                if F[idx_min] < f_best:
                    f_best = F[idx_min]
                    x_best = Y[idx_min].copy()

                # Sort
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]

                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

                # Update step size
                # Compute normalized step
                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.eye(dim)
                step = (L_inv @ (x_mean - x_old).T).T[0]
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-16:
                    ps = (1 - c_sigma) * ps + c_sigma * step_norm / chi_n * (step / step_norm)
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 100.0)

                # Update covariance matrix
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * (x_mean - x_old) / sigma

                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank_1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (Y_sorted[i] - x_old) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * (dim + 1 - c_1) - c_mu - c_c * (1 - c_c) * delta_hsig) * C \
                    + c_1 * (rank_1 - C) + c_mu * rank_mu

                # Ensure C is positive definite
                # Symmetrize
                C = (C + C.T) / 2
                # Check eigenvalues
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-10:
                        C = C + 1e-10 * np.eye(dim)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)

                gen += 1

                # Check if we should stop (convergence)
                if sigma < 1e-10 or (gen > 100 and np.ptp(F_sorted[:mu]) < 1e-8):
                    break

            return x_mean, f_best

        # Multi-start strategy
        n_restarts = min(5, budget // 100)
        evals_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0
            else:
                # Restart around current best with smaller sigma
                if x_best is not None:
                    x0 = np.clip(x_best + rng.standard_normal(dim) * 0.5, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.0

            rng_local = np.random.default_rng(42 + r)
            run_cmaes(x0, sigma0, evals_per_restart, rng_local)

        # Final polish with (1+1) if time remains
        if x_best is not None and time.time() < deadline - 5:
            x = x_best.copy()
            fx = f_best
            sigma = 0.5
            # Just do a few more steps
            for _ in range(50):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < f_best:
                        f_best, x_best = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return x_best, f_best

# EVOLVE-BLOCK-END
