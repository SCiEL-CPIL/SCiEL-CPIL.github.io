"""Rewritten optimizer: CMA-ES with multi-restart strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart strategy for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, budget_cma, n, rng):
            """Run CMA-ES from x0 with initial step sigma0."""
            x = x0.copy()
            sigma = sigma0
            fx = func(x)
            best_x = x.copy()
            best_fx = fx

            # CMA-ES parameters
            mu = max(1, n // 2)  # number of parents
            lam = 2 * mu  # population size
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            diag = np.ones(n)

            calls = 1  # already evaluated x0

            while calls < budget_cma and time_ok():
                # Generate population
                Y = rng.standard_normal((lam, n))
                X = x + sigma * (Y @ B * D)
                F = np.array([func(xi) for xi in X])
                calls += lam

                # Sort by fitness
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]

                # Update best
                if F[0] < best_fx:
                    best_fx = F[0]
                    best_x = X[order[0]].copy()

                # Weighted mean
                m = np.sum(weights[:, None] * Y[:mu], axis=0)

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D * m))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lam)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * m

                # Update covariance matrix
                delta_d = (1 - hsig) * c_c * (2 - c_c) if hsig == 0 else 0
                C = (1 - c_1 - c_mu + delta_d) * C
                C += c_1 * (np.outer(pc, pc) - C)
                C += c_mu * np.sum(weights[:, None, None] * np.outer(Y[:mu] * D, Y[:mu] * D), axis=0)

                # Eigen decomposition (periodic to save time)
                if calls % max(1, n // 2) == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    D = np.sqrt(eigvals)
                    B = eigvecs

                # Update step size
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(max(sigma, 1e-10), 100.0)

                # Update mean
                x = x + np.sum(weights[:, None] * X[:mu], axis=0)

                # Boundary handling
                x = np.clip(x, -5, 5)

                # Check for convergence
                if sigma < 1e-10 or np.all(np.abs(x - best_x) < 1e-8):
                    break

            return best_x, best_fx, calls

        # Multi-restart strategy
        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Initial exploration with diverse starting points
        n_starts = min(5, budget // 30)
        starts = []

        # Center
        starts.append(np.zeros(n))
        # Random points
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, n))

        # Stratified sampling for better coverage
        for i in range(2):
            for j in range(2):
                pt = np.zeros(n)
                pt[i] = -3.0 if j == 0 else 3.0
                starts.append(pt)

        # Deduplicate and limit
        seen = set()
        unique_starts = []
        for s in starts:
            key = tuple(np.round(s, 1))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)
        starts = unique_starts[:n_starts]

        # Evaluate starting points and find best
        for s in starts:
            if not time_ok() or total_calls >= budget:
                break
            s = np.clip(s, -5, 5)
            fs = func(s)
            total_calls += 1
            if fs < best_fx:
                best_fx = fs
                best_x = s.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_fx = func(best_x)
            total_calls += 1

        # Phase 2: CMA-ES from best start
        remaining = budget - total_calls
        if remaining > 10 and time_ok():
            x_best, fx_best, used = cma_es(func, best_x, 2.0, remaining, n, rng)
            total_calls += used
            if fx_best < best_fx:
                best_fx = fx_best
                best_x = x_best

        # Phase 3: If time/calls remain, try a few more restarts from diverse locations
        remaining = budget - total_calls
        if remaining > 20 and time_ok():
            n_extra = min(3, remaining // 20)
            for i in range(n_extra):
                if not time_ok() or remaining - total_calls < 15:
                    break
                # Generate a new random start far from current best
                while True:
                    s = rng.uniform(-4.5, 4.5, n)
                    if np.linalg.norm(s - best_x) > 2.0:
                        break
                s = np.clip(s, -5, 5)
                fs = func(s)
                total_calls += 1
                remaining = budget - total_calls
                if fs < best_fx:
                    best_fx = fs
                    best_x = s.copy()

                if remaining > 10 and time_ok():
                    x_try, fx_try, used = cma_es(func, s, 1.5, min(remaining - 2, 30), n, rng)
                    total_calls += used
                    remaining = budget - total_calls
                    if fx_try < best_fx:
                        best_fx = fx_try
                        best_x = x_try

        # Final polish: local search around best point
        remaining = budget - total_calls
        if remaining > 5 and time_ok() and best_x is not None:
            # Simple coordinate descent polish
            sigma_polish = 0.1
            x_polish = best_x.copy()
            fx_polish = best_fx
            for _ in range(min(remaining - 1, 20)):
                if not time_ok():
                    break
                improved = False
                for d in range(n):
                    for sign in [1, -1]:
                        y = x_polish.copy()
                        y[d] += sign * sigma_polish
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        total_calls += 1
                        if fy < fx_polish:
                            x_polish = y
                            fx_polish = fy
                            improved = True
                if improved:
                    sigma_polish *= 0.8
                else:
                    sigma_polish *= 0.5
                    if sigma_polish < 1e-6:
                        break
            if fx_polish < best_fx:
                best_fx = fx_polish
                best_x = x_polish

        return best_x, best_fx
# EVOLVE-BLOCK-END
