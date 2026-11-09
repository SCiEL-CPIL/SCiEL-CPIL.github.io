"""Optimizer for BBOB f6 (Attractive sector) using multi-start CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES with explicit time deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def remaining_calls():
            return max(0, budget - calls_made)

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Multi-start CMA-ES
        n_restarts = 5
        budgets_per_restart = []
        remaining = budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                b = remaining
            else:
                b = max(30, int(remaining * 0.35))
                b = min(b, remaining)
            budgets_per_restart.append(b)
            remaining -= b
        if sum(budgets_per_restart) > budget:
            budgets_per_restart[-1] = max(10, budget - sum(budgets_per_restart[:-1]))

        for r in range(n_restarts):
            if not time_ok() or remaining_calls() <= 0:
                break

            alloc = min(budgets_per_restart[r], remaining_calls())
            if alloc < 15:
                break

            # Initial point
            if r == 0:
                x0 = rng.uniform(-1, 1, dim)
            elif best_x is not None and rng.random() < 0.5:
                x0 = best_x + rng.standard_normal(dim) * 1.5
            else:
                x0 = rng.uniform(-3, 3, dim)
            x0 = np.clip(x0, -5, 5)

            # CMA-ES parameters
            n = dim
            popsize = max(4, int(4 + 3 * np.log(n)))
            mu = popsize // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, popsize + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights[:mu].sum()
            cc = (4. + mueff / (n + 4.)) / ((n + 2.) ** 2. + mueff + 2.)
            cs = (mueff + 2.) / (n + mueff + 5.)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2. + 1. / mueff) / ((n + 2.) ** 2 + mueff))

            mean = x0.copy()
            sigma = 2.0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            csqrt = 1.0 + (2.0 * (mueff - 2.0) / (n + 2.0)) * cs
            hsig = (1.5 + 1.0 / cs) * np.sqrt((n + 1) * (2 - 1.0 / (n + 1)))

            # Precompute Cholesky of C initially
            C_chol = np.linalg.cholesky(C)

            for gen in range(alloc):
                if not time_ok() or remaining_calls() <= 0:
                    break

                # Generate and evaluate population
                pop = np.zeros((popsize, n))
                fvals = np.zeros(popsize)
                for i in range(popsize):
                    if not time_ok() or remaining_calls() <= 0:
                        break
                    z = rng.standard_normal(n)
                    y = C_chol @ z
                    xi = mean + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fvals[i] = evaluate(xi)
                    pop[i] = xi

                if calls_made >= budget:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Weighted mean of best mu
                w = weights[:mu]
                new_mean = np.dot(w, pop_sorted[:mu])

                # Update evolution path for covariance (ps)
                old_mean = mean.copy()
                mean = new_mean

                # ps update
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_chol.T @ (new_mean - old_mean))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * hsig:
                    # Update pc
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (new_mean - old_mean) / sigma
                    C = (1 + c1 * cmu * mueff - c1) * C + c1 * np.outer(pc, pc) + cmu * np.dot(w, (pop_sorted[:mu] - old_mean) / sigma[:, None]) @ ((pop_sorted[:mu] - old_mean) / sigma[:, None]).T
                    C = (C + C.T) / 2  # symmetrize
                    # Update sigma
                    sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / hsig - 1))
                else:
                    C = (1 + c1 * cmu * mueff - c1) * C + cmu * np.dot(w, (pop_sorted[:mu] - old_mean) / sigma[:, None]) @ ((pop_sorted[:mu] - old_mean) / sigma[:, None]).T
                    C = (C + C.T) / 2

                # Ensure C is positive definite
                try:
                    C_chol = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Add small regularization
                    C += 1e-6 * np.eye(n)
                    C = (C + C.T) / 2
                    try:
                        C_chol = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        C_chol = np.eye(n)

                # Clamp sigma
                sigma = np.clip(sigma, 1e-4, 10.0)

                # Check convergence
                if sigma < 1e-4 or np.all(np.abs(C - np.eye(n)) < 1e-6):
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
