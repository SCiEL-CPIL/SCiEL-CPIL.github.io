"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start restarts for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, fx0, budget, rng, deadline):
        """Run CMA-ES from x0 for at most `budget` function calls."""
        n = self.dim
        m = max(4, int(4 + 2 * int(np.sqrt(n))))  # population size
        # Initial sigma based on box size
        sigma = 1.0
        # Covariance matrix
        C = np.eye(n)
        # Adaptation parameters
        cs = (m / 2 + 1) / (n + m / 2 + 2)
        ds = 1 + 2 * max(0, (np.sqrt((m + 1) / (n + 1)) - 1) / (n + 2)) + cs
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 1.7 + m + 2)
        cmu = min(1 - c1, (2 * (mu_w - 1) / ((n + 2) ** 2 + 1) - 1))
        # Weights
        weights = np.log(np.arange(1, m + 1))
        weights = weights / weights.sum()
        mu_w = int(np.sum(weights[:m] > 0))
        mu_w = min(mu_w, m)
        weights = weights[:mu_w]
        weights = weights / weights.sum()
        mu_w = int(np.sum(weights > 0))
        lambda_w = m
        # Recalculate
        weights = np.log(np.arange(1, m + 1))
        weights = weights / weights.sum()
        mu_w = min(int(np.sum(weights > 0)), m)
        weights = weights[:mu_w]
        weights = weights / weights.sum()
        mu_w = int(np.sum(weights > 0))
        # Effective population size
        mueff = 1.0 / np.sum(weights ** 2)
        # Recalculate c1, cmu with correct mueff
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 1.7 + m + 2)
        cmu = min(1 - c1, (2 * (mueff - 1) / ((n + 2) ** 2 + 1) - 1))
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        # Initial eigenvalue decomposition
        D = np.ones(n)
        B = np.eye(n)
        # Track best
        x_best = x0.copy()
        f_best = fx0
        calls = 1
        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            # Generate population
            ys = []
            for i in range(m):
                z = rng.standard_normal(n)
                y = x_best + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                ys.append(y)
            # Evaluate
            fvals = []
            for y in ys:
                if calls >= budget:
                    break
                fy = func(y)
                fvals.append(fy)
                calls += 1
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
            if len(fvals) < m:
                break
            fvals = np.array(fvals)
            # Sort
            idx = np.argsort(fvals)
            ys_sorted = [ys[i] for i in idx]
            fvals_sorted = fvals[idx]
            # Weighted mean
            ymean = sum(weights[i] * ys_sorted[i] for i in range(mu_w))
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D * ((ymean - x_best) / sigma)))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * calls)) < (1.4 + 2 / (n + 1)) * c_sigma:
                pc = (1 - cs) * pc + cs * (ymean - x_best) / sigma
                C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - (1 - cs) ** 2) * C) + cmu * sum(
                    weights[i] * (np.outer(ys_sorted[i] - ymean, ys_sorted[i] - ymean)) for i in range(mu_w)
                )
            else:
                pc = (1 - cs) * pc
                C = (1 - c1 - cmu) * C + c1 * ((1 - (1 - cs) ** 2) * C) + cmu * sum(
                    weights[i] * (np.outer(ys_sorted[i] - ymean, ys_sorted[i] - ymean)) for i in range(mu_w)
                )
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / c_sigma - 1))
            # Update mean
            x_best = ymean
            # Ensure symmetry
            C = (C + C.T) / 2
            # Eigenvalue decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                D = np.ones(n)
                B = np.eye(n)
            # Adaptive sigma
            if sigma > 50:
                sigma = 50
            if sigma < 1e-8:
                sigma = 1e-8
        return x_best, f_best, calls

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        rng = np.random.default_rng(0)

        # Multi-start: split budget among several restarts
        # Use 3 restarts with decreasing budgets
        n_restarts = 3
        # First restart gets 50%, second 30%, third 20%
        budgets = [int(total_budget * 0.5), int(total_budget * 0.3), int(total_budget * 0.2)]
        # Adjust to sum to total_budget
        diff = total_budget - sum(budgets)
        budgets[0] += diff

        x_best_global = None
        f_best_global = np.inf
        calls_used = 0

        for r in range(n_restarts):
            if calls_used >= total_budget:
                break
            if time.time() > deadline:
                break
            # Choose starting point
            if r == 0:
                x0 = rng.uniform(-5, 5, n)
            else:
                # Start near best found so far with some perturbation
                if x_best_global is not None:
                    x0 = np.clip(x_best_global + rng.standard_normal(n) * 2.0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, n)
            fx0 = func(x0)
            calls_used += 1
            if fx0 < f_best_global:
                f_best_global = fx0
                x_best_global = x0.copy()
            # Run CMA-ES
            x_res, f_res, c = self._cma_es(func, x0, fx0, budgets[r], rng, deadline)
            calls_used += c - 1  # -1 because fx0 already counted
            if f_res < f_best_global:
                f_best_global = f_res
                x_best_global = x_res.copy()

        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
