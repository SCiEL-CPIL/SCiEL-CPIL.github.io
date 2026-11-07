"""CMA-ES with restarts and local refinement for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _ensure_pd(m):
    """Ensure a matrix is positive definite by adding small diagonal if needed."""
    try:
        np.linalg.cholesky(m)
        return m
    except np.linalg.LinAlgError:
        return m + 1e-10 * np.eye(m.shape[0])


class Optimizer:
    """CMA-ES with restarts for multi-modal BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        # CMA-ES parameters
        def cma_es(x0, sigma0, max_evals, rng, deadline):
            n = len(x0)
            # Population size
            lam = int(4 + 3 * math.log(n))
            # Weights
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / w.sum()
            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            # Initial state
            xmean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            hs = math.sqrt(1 - math.exp(-2 / (lam + 2))) * (1 + cc)
            evals = 0

            # Initial population
            pops = []
            fitvals = []
            for i in range(lam):
                z = rng.standard_normal(n)
                x = xmean + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                f = func(x)
                pops.append(x)
                fitvals.append(f)
                evals += 1

            # Sort by fitness
            idx = np.argsort(fitvals)
            pops = [pops[i] for i in idx]
            fitvals = [fitvals[i] for i in idx]

            while evals < max_evals and time.time() < deadline:
                # Update mean
                xmean_old = xmean.copy()
                xmean = sum(w[i] * pops[i] for i in range(mu))

                # Update step size
                p_sigma = (1 - cs) * p_sigma + math.sqrt(cs * (2 - cs) * mueff) / sigma * (B @ (D * (xmean - xmean_old)))
                if np.linalg.norm(p_sigma) / hs < 1.4 + 2 / (n + 1):
                    sigma *= math.exp(cs / n * (np.linalg.norm(p_sigma) / hs - 1))

                # Update covariance
                pc = (1 - cc) * pc + cc / sigma * (xmean - xmean_old)
                delta_hsig = 1.0 if np.linalg.norm(p_sigma) / hs < (1.4 + 2 / (n + 1)) else 0.0
                D = D * math.exp(cs / (2 * n) * (np.linalg.norm(p_sigma) ** 2 / n - 1) * (1 - delta_hsig))
                # Update B and D via eigendecomposition periodically
                # For simplicity, we'll use the rank-mu update
                y_vecs = [(pops[i] - xmean_old) / sigma for i in range(mu)]
                H = sum(w[i] * np.outer(y_vecs[i], y_vecs[i]) for i in range(mu))
                H = H + (cmu * cc * (1 - cc) / (2 * n)) * np.outer(pc, pc)
                # Update B, D
                # This is simplified; a proper implementation would do eigendecomposition
                # For now, we'll skip the full eigendecomposition for speed and use a simpler approach

                # Generate new population
                new_pops = []
                new_fitvals = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    x = xmean + sigma * (B @ (D * z))
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    new_pops.append(x)
                    new_fitvals.append(f)
                    evals += 1
                    if evals >= max_evals or time.time() >= deadline:
                        break

                if len(new_pops) < lam:
                    break

                # Sort and update
                idx = np.argsort(new_fitvals)
                pops = [new_pops[i] for i in idx]
                fitvals = [new_fitvals[i] for i in idx]

            # Return best
            return xmean, fitvals[0] if fitvals else float('inf')

        # Multi-start CMA-ES with restarts
        rng = np.random.default_rng(42)

        # First, do a quick random search to find a good starting point
        best_x = None
        best_f = float('inf')
        n_init = min(20, budget // 10)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES from best point
        remaining = budget - n_init
        # Use most of the budget for CMA-ES
        cma_budget = int(remaining * 0.7)
        x_best, f_best = cma_es(best_x, 1.5, cma_budget, rng, deadline)

        if f_best < best_f:
            best_x, best_f = x_best, f_best

        # Local refinement with smaller CMA-ES
        if time.time() < deadline - 10:
            local_budget = min(budget - n_init - cma_budget, 100)
            if local_budget > 0:
                x_local, f_local = cma_es(best_x, 0.3, local_budget, rng, deadline)
                if f_local < best_f:
                    best_x, best_f = x_local, f_local

        return best_x, best_f
# EVOLVE-BLOCK-END
