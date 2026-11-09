"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive population-based optimizer with restarts and CMA-like covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)

    def _run_cma(self, func, x0, budget, rng, deadline, sigma0=1.0):
        """Run a CMA-ES-like optimizer with adaptive covariance."""
        n = self.dim
        lam = max(4, 2 * n + 2)  # population size
        c = 1.0 / (n + 4)
        d = 1.0 + 2 * max(0, np.sqrt((lam // 2 - 1) / (n + 1)) - 1)
        cm = 1 + c * d + 2 * c / (n + 1)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_w = np.sum(w[:mu])
        c_sigma = (mu_w + 2) / (n + mu_w / 2)
        c_cov = (2 + mu_w / n) / (n + 2 + mu_w / n)

        C = np.eye(n)
        sqrtC = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        x = np.array(x0, dtype=float)
        sigma = sigma0
        best_x = x.copy()
        best_f = func(x)

        calls = 1
        t_start = time.time()

        for gen in range(budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                # Approximate sqrt(C) * z using current B and D
                y = x + sigma * (B @ (D * z))
                y = self._clip(y)
                offspring.append(y)

            # Evaluate
            fvals = []
            for y in offspring:
                if calls >= budget:
                    break
                fy = func(y)
                fvals.append(fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(fvals) < lam:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            fvals_sorted = np.array(fvals)[idx]
            ys_sorted = [offspring[i] for i in idx]

            # Update mean
            x_new = np.sum([w[i] * ys_sorted[i] for i in range(mu)], axis=0)
            x_new = self._clip(x_new)

            # Update evolution paths
            d_mean = (x_new - x) / sigma
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * d_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * np.linalg.inv(sqrtC) @ d_mean

            # Update covariance
            delta_hs = (np.linalg.norm(ps) > (1.4 + 1 / n) * chi_n)
            h1 = (1 - delta_hs)
            C = (1 - c_cov) * C + c_cov * (
                np.outer(pc, pc) * h1 +
                np.sum([w[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu)]) / sigma**2
            )
            C = (C + C.T) / 2

            # Update sqrt(C) via eigen decomposition (approximate for small n)
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            D = np.sqrt(eigenvalues)
            B = eigenvectors
            sqrtC = B @ np.diag(D)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

            x = x_new

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Deadline: leave 5 seconds margin
        t_start = time.time()
        deadline = t_start + 290.0

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Multi-start with diverse initializations
        # Use a few restarts with different strategies

        # Initial random points for diversity
        n_starts = min(5, budget // 50)
        remaining_budget = budget

        for start_i in range(n_starts):
            if total_calls >= budget or time.time() > deadline:
                break

            if start_i == 0:
                x0 = np.zeros(n)
            elif start_i == 1:
                x0 = rng.uniform(-4, 4, n)
            elif start_i == 2:
                x0 = rng.uniform(-5, 5, n)
            else:
                # Perturb around current best
                if best_x is not None:
                    x0 = self._clip(best_x + 1.5 * rng.standard_normal(n))
                else:
                    x0 = rng.uniform(-5, 5, n)

            # Allocate budget to this run
            alloc = max(10, remaining_budget // (n_starts - start_i))
            alloc = min(alloc, budget - total_calls)

            x_best, f_best, calls_used = self._run_cma(
                func, x0, alloc, rng, deadline, sigma0=2.0
            )

            total_calls += calls_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

            remaining_budget = budget - total_calls

        # Phase 2: Final refinement around best with remaining budget
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > 5:
                x_final, f_final, calls_final = self._run_cma(
                    func, best_x, remaining, rng, deadline, sigma0=0.5
                )
                total_calls += calls_final
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final

        if best_x is None:
            # Fallback: at least one call
            x = rng.uniform(-5, 5, n)
            best_f = func(x)
            best_x = x
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
