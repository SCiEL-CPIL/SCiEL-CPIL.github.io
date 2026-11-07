"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + 3 * ln(n)
        lam = max(4 + int(3 * np.log(dim)), 4)
        # Initial sigma: fraction of domain half-width
        sigma = 2.0  # domain is [-5,5], so half-width is 5
        # Step size for mean update
        cm = 1 + 2 / (dim + 1) ** 1.8 - 1 / lam
        cs = (dim + 1.3) / (dim + lam + 2.5)
        ds = 1 + 2 * max(0, np.sqrt((cs + 1) / (dim + 1) - 1)) + cs
        cc = (4 - dim + 1 / dim) / (dim + 4 / 3)
        cmu = 2 / ((dim + 1.3) ** 2 + lam)

        def cma_es(x0, sigma0, eval_count_remaining, rng):
            x_mean = x0.copy().astype(float)
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            best_x = x_mean.copy()
            best_f = func(best_x)
            evals_used = 1

            while evals_used < eval_count_remaining:
                if time.time() > deadline:
                    break

                # Generate population
                z = rng.standard_normal((lam, dim))
                # Whitening transform: C = B @ B.T, so B is Cholesky
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.eye(dim)

                pops = np.array([x_mean + sigma * (B @ z[i]) for i in range(lam)])
                pops = np.clip(pops, -5, 5)

                # Evaluate
                fvals = np.array([func(pops[i]) for i in range(lam)])
                evals_used += lam

                # Track best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = pops[idx_min].copy()

                # Sort by fitness
                order = np.argsort(fvals)
                z_sorted = z[order]
                f_sorted = fvals[order]

                # Weights for mean update
                mu = lam // 2
                w = np.zeros(mu)
                w[0] = np.log(mu + 0.5) - np.mean(np.log(np.arange(1, mu + 1)))
                w = w / np.sum(w)
                mueff = 1 / np.sum(w ** 2)

                # Update mean
                x_new = x_mean + sigma * sum(w[i] * z_sorted[i] for i in range(mu))

                # Update evolution paths
                pc = (1 - cm) * pc + cm * (x_new - x_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (x_new - x_mean) / sigma

                # Update covariance
                delta_hs = (np.sqrt((1 - (1 - cs) ** 2)) - (1 - cs)) / (cs * dim)
                rank_mu = (C - np.outer(pc, pc)) / dim + delta_hs * np.outer(pc, pc)
                C = (1 - (cmu + cm) / dim) * C + cmu * sum(w[i] ** 2 * np.outer(z_sorted[i], z_sorted[i]) for i in range(mu)) + cm * rank_mu

                # Symmetrize
                C = (C + C.T) / 2

                # Adapt sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 50.0))

                x_mean = x_new

            return best_x, best_f, evals_used

        # Phase 1: Initial exploration with a few random starts
        rng = np.random.default_rng(42)

        # Try a few different initial points to handle the hidden shift
        # Use a small number of restarts
        n_restarts = 3
        evals_per_restart = budget // n_restarts

        best_x = None
        best_f = np.inf
        total_evals = 0

        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if total_evals >= budget:
                break

            # Different initializations for each restart
            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 3.0
            elif r == 1:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 2.0
            else:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.5

            remaining = budget - total_evals
            if remaining <= 0:
                break

            x_best, f_best, evals_used = cma_es(x0, sigma0, remaining, rng)
            total_evals += evals_used

            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        # Phase 2: Coordinate descent polish on the best point
        if best_x is not None and total_evals < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            # Try small perturbations in each coordinate
            for coord in range(dim):
                if time.time() > deadline or total_evals >= budget:
                    break
                step = 0.5
                for _ in range(50):
                    if time.time() > deadline or total_evals >= budget:
                        break
                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[coord] = min(5.0, x_trial[coord] + step)
                    if total_evals < budget:
                        ft = func(x_trial)
                        total_evals += 1
                        if ft < fx:
                            x, fx = x_trial, ft
                        else:
                            step *= 0.5
                            if step < 1e-4:
                                break
                    # Try negative direction
                    x_trial = x.copy()
                    x_trial[coord] = max(-5.0, x_trial[coord] - step)
                    if total_evals < budget:
                        ft = func(x_trial)
                        total_evals += 1
                        if ft < fx:
                            x, fx = x_trial, ft
                        else:
                            step *= 0.5
                            if step < 1e-4:
                                break

            if fx < best_f:
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
