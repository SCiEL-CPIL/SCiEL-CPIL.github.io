"""Black-box continuous optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES with adaptive restarts and time-aware budget management.
    Uses a population-based approach with covariance matrix adaptation for
    robust optimization of the smooth Discus landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Global best tracking
        best_x = None
        best_f = np.inf

        # Multi-start with adaptive restarts
        # Use a portion of budget per restart, increasing as we go
        # Start with larger populations for exploration, smaller for exploitation

        # Phase 1: Broad exploration with several short CMA-ES runs
        # Phase 2: Focused exploitation around best found

        # We'll do a sequence of CMA-ES runs with different initializations
        # and step sizes, keeping the best overall.

        # Configuration for CMA-ES
        def cma_es(func, x0, sigma0, max_evals, rng, deadline, best_state):
            """
            Run CMA-ES from x0 with initial sigma0 for at most max_evals evaluations.
            Returns (best_x, best_f, evals_used).
            """
            n = len(x0)
            x = np.array(x0, dtype=float)
            sigma = sigma0
            best_x = x.copy()
            best_f = func(x)
            evals = 1
            best_state[0] = best_x.copy()
            best_state[1] = best_f

            if max_evals <= 1:
                return best_x, best_f, evals

            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lambda_ // 2  # number of parents
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # Step size control
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            hsig = 1 - np.exp(-1 / (budget * damps))

            # State variables
            ps = np.zeros(n)
            pc = np.zeros((n, n))
            C = np.eye(n)
            Cinv = np.eye(n)
            invsqrtC = np.eye(n)
            invsqrtC = np.linalg.cholesky(C)  # Lower triangular
            # Actually, we need to maintain Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except:
                L = np.eye(n)
            invsqrtC = np.linalg.inv(L)

            # Cumulative step size
            sigma = sigma0

            while evals < max_evals:
                # Check time
                if time.time() > deadline:
                    break

                # Generate population
                pop_x = []
                pop_f = []
                z = rng.standard_normal((lambda_, n))
                for i in range(lambda_):
                    y = x + sigma * (invsqrtC @ z[i])
                    y = np.clip(y, -5, 5)
                    pop_x.append(y)
                    pop_f.append(func(y))
                    evals += 1
                    if time.time() > deadline:
                        break

                if evals >= max_evals:
                    break

                # Sort by fitness
                order = np.argsort(pop_f)
                pop_x_sorted = [pop_x[i] for i in order]
                pop_f_sorted = [pop_f[i] for i in order]

                # Update best
                if pop_f_sorted[0] < best_f:
                    best_f = pop_f_sorted[0]
                    best_x = pop_x_sorted[0].copy()
                    best_state[0] = best_x.copy()
                    best_state[1] = best_f

                # Update mean
                x_old = x.copy()
                x = sum(weights[i] * pop_x_sorted[i] for i in range(mu))
                x = np.clip(x, -5, 5)

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ ((x - x_old) / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (max_evals - evals))) < (1.4 + 2 / (n + 1)) * chi_n:
                    hsig = 1.0
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update covariance matrix
                v = (x - x_old) / sigma
                pc = (1 - cs) * pc + hsig * cs * (2 - cs) * np.outer(v, v)
                delta_d = hsig * np.sqrt(cs * (2 - cs) * mueff) * (np.linalg.norm(v) / chi_n - 1)
                C = (1 + c1 * delta_d * delta_d + cmu) * C \
                    - c1 * pc - cmu * sum(weights[i] * np.outer(pop_x_sorted[i] - x_old, pop_x_sorted[i] - x_old) for i in range(mu))

                # Ensure positive definiteness
                try:
                    L = np.linalg.cholesky(C)
                except:
                    # Regularize
                    C += 1e-6 * np.eye(n)
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        C = np.eye(n)
                        L = np.eye(n)

                invsqrtC = np.linalg.inv(L)

                # Check convergence
                if sigma < 1e-8 or (np.max(C) / np.min(np.diag(C)) > 1e10):
                    break

            return best_x, best_f, evals

        # Strategy: Multiple restarts with different initializations
        # Use a fraction of budget for each restart, adapting based on progress

        # Initial broad search
        best_state = [None, np.inf]
        remaining = budget
        num_restarts = 0
        min_budget_per_restart = 30

        # First: a few quick exploratory runs from different points
        # Then: focused runs around the best found

        # Phase 1: Exploration - 3-5 short runs from random points
        n_explore = 4
        explore_budget = max(min_budget_per_restart, budget // (n_explore + 2))

        for i in range(n_explore):
            if remaining <= min_budget_per_restart:
                break
            if time.time() > deadline:
                break

            # Random initialization
            x0 = rng.uniform(-5, 5, n)
            sigma0 = 2.0

            alloc = min(explore_budget, remaining)
            bx, bf, used = cma_es(func, x0, sigma0, alloc, rng, deadline, best_state)
            remaining -= used
            num_restarts += 1

        # Phase 2: Exploitation - longer runs around best found
        if remaining > min_budget_per_restart and best_state[0] is not None:
            # Run a longer CMA-ES from the best point
            x0 = best_state[0].copy()
            sigma0 = 0.5  # Smaller step for exploitation
            alloc = min(remaining, budget // 2)
            bx, bf, used = cma_es(func, x0, sigma0, alloc, rng, deadline, best_state)
            remaining -= used
            num_restarts += 1

        # Phase 3: If budget remains, do a few more focused runs
        while remaining > min_budget_per_restart and time.time() < deadline:
            if best_state[0] is not None:
                # Perturb the best point
                x0 = best_state[0] + 0.5 * rng.standard_normal(n)
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = rng.uniform(-5, 5, n)

            sigma0 = 0.3
            alloc = min(remaining, 50)
            bx, bf, used = cma_es(func, x0, sigma0, alloc, rng, deadline, best_state)
            remaining -= used
            num_restarts += 1

            if remaining <= min_budget_per_restart:
                break

        # Final: Use any remaining budget with a very focused search
        if remaining > 5 and best_state[0] is not None:
            x0 = best_state[0].copy()
            sigma0 = 0.1
            alloc = remaining
            bx, bf, used = cma_es(func, x0, sigma0, alloc, rng, deadline, best_state)
            remaining -= used

        return best_state[0], best_state[1]
# EVOLVE-BLOCK-END
