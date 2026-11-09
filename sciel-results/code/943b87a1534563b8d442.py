"""CMA-ES optimizer for the BBOB f11 (Discus) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts on stagnation, budget- and time-aware."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s / 330s kill

        # Global best across all restarts
        global_x = None
        global_f = np.inf

        # CMA-ES hyperparameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        # Weights
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # Step size adaptation
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 2)) + cs
        # Covariance adaptation
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Damping
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def run_cmaes(x0, f0, remaining_budget):
            nonlocal global_x, global_f
            xmean = np.array(x0, dtype=float)
            fmean = f0
            sigma = 2.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            calls = 1  # f0 already counted
            gen = 0
            best_f = f0
            best_x = np.array(x0, dtype=float)
            no_improve = 0

            while calls < remaining_budget:
                if time.time() > deadline:
                    break
                # Sample population
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                ys = rng.standard_normal((lambda_, dim))
                # Transform: x = xmean + sigma * B @ (D * y)
                pops = xmean + sigma * (B @ (D * ys.T)).T
                # Clip to box
                pops = np.clip(pops, -5, 5)
                # Evaluate
                fvals = np.empty(lambda_)
                for i in range(lambda_):
                    if time.time() > deadline:
                        fvals[i] = np.inf
                        continue
                    fvals[i] = func(pops[i])
                    calls += 1
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pops[i].copy()
                        no_improve = 0
                        if fvals[i] < global_f:
                            global_f = fvals[i]
                            global_x = pops[i].copy()
                    else:
                        no_improve += 1
                    if calls >= remaining_budget:
                        break

                if calls >= remaining_budget:
                    break

                # Sort
                order = np.argsort(fvals)
                ys_sorted = ys[order]
                fvals_sorted = fvals[order]
                xs_sorted = pops[order]

                # Update mean
                xmean_new = xmean + sigma * (B @ (D * (w @ ys_sorted))).T
                xmean_new = np.clip(xmean_new, -5, 5)

                # Update step size
                z = (B @ (D * (xmean_new - xmean) / sigma)) / sigma if sigma > 1e-12 else np.zeros(dim)
                # More carefully: z = (xmean_new - xmean) / sigma in the standardized space
                z = (xmean_new - xmean) / (sigma + 1e-12)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1)) / chiN < (1.4 + 2 / (dim + 1)) * (1 - cs))
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 20.0)

                # Update covariance
                v = (xmean_new - xmean) / (sigma + 1e-12)
                rank_one = np.outer(v, v)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    yi = ys_sorted[i]
                    xi = (xs_sorted[i] - xmean) / (sigma + 1e-12)
                    rank_mu += w[i] * np.outer(xi, xi)
                C = (1 + c1 * hsig * (cs * (2 - cs)) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                # Symmetrize
                C = (C + C.T) / 2

                xmean = xmean_new
                gen += 1

                # Restart condition: condition number too high or no improvement for many gens
                evals_used = gen * lambda_
                if evals_used > 10 * dim * dim and no_improve > 15:
                    break
                if evals_used > 5 * dim * dim:
                    try:
                        eigvals = np.linalg.eigvalsh(C)
                        cond = eigvals[-1] / max(eigvals[0], 1e-12)
                        if cond > 1e12:
                            break
                    except np.linalg.LinAlgError:
                        break

            return best_x, best_f, calls

        rng = np.random.default_rng(42)

        # Initial point: center
        x0 = np.zeros(dim)
        f0 = func(x0)
        if f0 < global_f:
            global_f = f0
            global_x = x0.copy()
        calls_used = 1

        # Run CMA-ES with the full remaining budget; restarts handled internally
        # We do a single run but allow internal restart via re-initialization
        # For robustness across the 3 instances, we rely on the single CMA-ES run
        # with adaptive covariance. If it stagnates, we do a few random restarts.
        x_best, f_best, calls_this_run = run_cmaes(x0, f0, budget)
        calls_used += calls_this_run - 1  # f0 already counted

        # If we still have budget and time, do a couple of random restarts
        restarts = 0
        while calls_used < budget and time.time() < deadline and restarts < 3:
            if global_f < 1e-6:
                break
            x0r = rng.uniform(-4, 4, dim)
            f0r = func(x0r)
            calls_used += 1
            if f0r < global_f:
                global_f = f0r
                global_x = x0r.copy()
            # Short CMA-ES run from this point
            remaining = budget - calls_used
            if remaining < lambda_ + 5:
                break
            x_best_r, f_best_r, calls_r = run_cmaes(x0r, f0r, remaining + calls_used)
            calls_used += calls_r - 1
            restarts += 1
            if time.time() > deadline - 5:
                break

        if global_x is None:
            global_x = np.zeros(dim)
        return global_x, global_f
# EVOLVE-BLOCK-END
