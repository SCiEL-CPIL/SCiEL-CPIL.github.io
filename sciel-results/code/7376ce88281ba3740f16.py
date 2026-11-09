"""Multi-restart CMA-ES with deadline guard for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Global best tracking
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES implementation (lightweight, no scipy dependency)
        def cma_es(func, x0, sigma0, max_evals, deadline):
            nonlocal best_x, best_f
            dim = self.dim
            rng = np.random.default_rng()

            # Initial parameters
            x = x0.copy()
            fx = func(x)
            update_best(x, fx)
            evals = 1

            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            cc = (4 + dim / 3) / ((dim + 4) ** 2 + 2)
            cs = (2 + 2 / (dim + 1)) / ((dim + 1) ** 2 + 2)
            cmu = min(0.2, (2 * (2 + dim / (dim + 2)) - 1) / ((dim + 2) ** 2 + 2))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Population size
            lambda_ = 4 + int(3 * np.log(dim))
            mu = lambda_ // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / (weights ** 2).sum()

            # Loop
            while evals < max_evals and time.time() < deadline:
                # Generate population
                D = np.sqrt(np.diag(C))
                B = np.linalg.inv(np.sqrtm(C + 1e-10 * np.eye(dim))) if False else None
                # Use Cholesky-like approach: C = B B^T
                try:
                    L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                offspring = []
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals += 1
                    offspring.append((fy, y))

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                y_sorted = [o[1] for o in offspring]
                f_sorted = [o[0] for o in offspring]

                # Update best
                for i in range(min(mu, lambda_)):
                    update_best(y_sorted[i], f_sorted[i])

                # Compute weighted mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * y_sorted[i]
                x_new = y_w

                # Adaptation
                # Update C
                diff = (y_sorted[:mu] - x).T
                C = (1 - cs * (2 - cs) - cmu) * C
                C += cs * (2 - cs) * np.outer(pc, pc)
                for i in range(mu):
                    C += cmu * weights[i] * np.outer(diff[:, i], diff[:, i])

                # Update pc
                pc = (1 - cc) * pc + cc * (x_new - x) / sigma

                # Update sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (diff @ weights) / sigma
                sigma *= np.exp((1 / 5) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100)

                x = x_new
                fx = f_sorted[0]

                # Check if converged
                if np.linalg.norm(x_new - x) < 1e-10:
                    break

            return x, fx

        # Strategy: multiple restarts with diverse starting points
        rng = np.random.default_rng(0)

        # Create diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, dim))
        # Axis-aligned points
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 3.0
            starts.append(pt)
            pt[i] = -3.0
            starts.append(pt)

        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        # Budget allocation: first restart gets more, subsequent get less
        n_restarts = min(len(starts), 8)
        # Allocate: 40%, 25%, 15%, 10%, 5%, 3%, 1%, 1%
        allocs = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.01, 0.01]
        allocs = allocs[:n_restarts]
        # Normalize to sum to 1
        total = sum(allocs)
        allocs = [a / total for a in allocs]

        remaining = budget
        for i in range(n_restarts):
            if time.time() >= deadline:
                break
            alloc = int(allocs[i] * budget)
            if i == n_restarts - 1:
                alloc = remaining  # Use all remaining
            if alloc < 10:
                alloc = min(remaining, 10)
            if alloc <= 0:
                break

            x0 = starts[i]
            sigma0 = 2.0 if i == 0 else 1.5
            cma_es(func, x0, sigma0, alloc, deadline)
            remaining -= alloc

        return best_x, best_f
# EVOLVE-BLOCK-END
