"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        total_budget = self.budget
        dim = self.dim

        # Use multiple restarts of CMA-ES
        # Each restart gets a portion of the budget
        # First restart: larger budget for initial exploration
        # Subsequent restarts: smaller budgets for refinement

        # Strategy: 
        # - Initial phase: 1 CMA-ES run with ~60% budget
        # - Then several shorter CMA-ES runs from different starts

        # Actually, let's do a single well-tuned CMA-ES with population size
        # and proper stopping criteria, plus a few restarts if budget allows

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, budget, deadline, best_x, best_f):
            """Run CMA-ES for given budget, return (best_x, best_f, calls_made)."""
            n = self.dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            # Standard CMA-ES parameters
            mu = lam // 2  # number of recombination weights
            # Weights: linearly decreasing
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            # Step size control parameters
            cs = (mueff + 1) / (n + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) - cs
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cmu1 = 2 * (mueff + 1.5) / ((n + 2) ** 2 + mueff + 1.5)
            cmu2 = min(1 - cmu1, 2 * (mueff - 1 + 1 / mueff) / ((n + 2) ** 2 + mueff + 1))
            # Initial covariance
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            sigma = sigma0
            x = x0.copy()
            fx = func(x)
            calls = 1
            best_local_x = x.copy()
            best_local_f = fx

            for gen in range(budget // lam + 1):
                if time.time() > deadline:
                    break
                # Sample population
                ys = np.zeros((lam, n))
                for i in range(lam):
                    z = np.random.standard_normal(n)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    ys[i] = y
                    fy = func(y)
                    calls += 1
                    if fy < best_local_f:
                        best_local_f = fy
                        best_local_x = y.copy()
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                if time.time() > deadline:
                    break

                # Sort by fitness
                fvals = np.array([func(ys[i]) for i in range(lam)])  # This is wrong - we already evaluated
                # Actually we need to store fvals during sampling
                # Let me fix this

                # Re-do: store fvals during sampling
                # Let me restructure

            return best_x, best_f, calls

        # Let me rewrite more carefully
        def cma_es_run(func, x0, sigma0, budget, deadline):
            """Run CMA-ES, return (best_x, best_f, calls_made)."""
            n = self.dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cs = (mueff + 1) / (n + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) - cs
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cmu1 = 2 * (mueff + 1.5) / ((n + 2) ** 2 + mueff + 1.5)
            cmu2 = min(1 - cmu1, 2 * (mueff - 1 + 1 / mueff) / ((n + 2) ** 2 + mueff + 1))
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            sigma = sigma0
            x = x0.copy()
            fx = func(x)
            calls = 1
            best_local_x = x.copy()
            best_local_f = fx

            for gen in range(1, budget // lam + 2):
                if time.time() > deadline or calls >= budget:
                    break

                # Sample and evaluate population
                ys = np.zeros((lam, n))
                fvals = np.zeros(lam)
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = np.random.standard_normal(n)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    ys[i] = y
                    fvals[i] = func(y)
                    calls += 1
                    if fvals[i] < best_local_f:
                        best_local_f = fvals[i]
                        best_local_x = y.copy()
                
                if calls >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                idx = np.argsort(fvals[:lam])
                ys = ys[idx]
                fvals = fvals[idx]

                # Mean of selected points
                x_new = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update evolution paths
                y_mean = (x_new - x) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_mean
                hsig = (np.linalg.norm(p_sigma) /
                        np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) /
                        np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))))
                # Simplified hsig
                hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
                hsig = min(hsig / np.sqrt(n) * (1 + 2 / (n + 1)), 1.4 + 2 / (n + 1))

                p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_mean

                # Update covariance
                if hsig < 1.4 + 2 / (n + 1):
                    delta = (1 - (1 - ds) ** (2 * (gen + 1))) * (1 - cc) * np.outer(p_c, p_c)
                    C = (1 - cmu1 - cmu2 + cc * (1 - cc) * (1 + ds) - (1 - ds) * cc) * C \
                        + cmu1 * np.outer(y_mean, y_mean) \
                        + cmu2 * np.sum(weights[:, None] * np.outer(ys[:mu], ys[:mu]), axis=0) \
                        + delta

                # Regularize C
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update C_inv_sqrt
                eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
                C_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

                # Update sigma
                sigma = sigma * np.exp(cs / (n + 1) * (np.linalg.norm(p_sigma) / np.sqrt(n) - 1))

                # Move mean
                x = x_new
                fx = fvals[0]

            return best_local_x, best_local_f, calls

        # Main strategy: multiple CMA-ES restarts
        # First, do a quick random search to find a good starting point
        # Then run CMA-ES from the best point

        # Phase 1: Random initialization with a few points
        rng = np.random.default_rng(42)
        n_init = 10
        init_x = rng.uniform(-5, 5, (n_init, dim))
        init_f = np.array([func(x) for x in init_x])
        calls = n_init

        best_idx = np.argmin(init_f)
        best_x = init_x[best_idx].copy()
        best_f = init_f[best_idx]

        # Phase 2: CMA-ES from the best initial point
        remaining = total_budget - calls
        if remaining > 0 and time.time() < deadline:
            sigma0 = 1.0
            bx, bf, bc = cma_es_run(func, best_x, sigma0, remaining, deadline)
            calls += bc
            if bf < best_f:
                best_f = bf
                best_x = bx

        # Phase 3: If budget remains, try another CMA-ES from a different start
        if calls < total_budget and time.time() < deadline:
            remaining = total_budget - calls
            if remaining > 20:
                # Try from a different random point
                x2 = rng.uniform(-5, 5, dim)
                sigma0 = 0.5
                bx, bf, bc = cma_es_run(func, x2, sigma0, remaining, deadline)
                calls += bc
                if bf < best_f:
                    best_f = bf
                    best_x = bx

        return best_x, best_f
# EVOLVE-BLOCK-END
