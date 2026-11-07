"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for the Bent Cigar function f12."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # lambda = population size
        lam = max(4, int(4 + 3 * np.log(n)))  # standard CMA-ES lambda
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc - 2 / max(n - 1, 1)) / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

        # Initial point: random in [-5, 5]
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_fx = fx
        best_x = x.copy()

        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls = 1

        # Reserve some budget for polish
        polish_budget = min(30, budget // 15)
        main_budget = budget - polish_budget

        while calls < main_budget:
            if time.time() > deadline:
                break

            # Generate population
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)

            # Generate lam random points
            pop = np.zeros((lam, n))
            pop_f = np.zeros(lam)
            for i in range(lam):
                z = rng.standard_normal(n)
                step = sigma * np.dot(L, z)
                y = np.clip(x + step, -5, 5)
                pop[i] = y
                pop_f[i] = func(y)
                calls += 1
                if calls >= budget:
                    break
                if pop_f[i] < best_fx:
                    best_fx = pop_f[i]
                    best_x = y.copy()
                if time.time() > deadline:
                    break

            if calls >= budget or time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(pop_f)
            pop = pop[idx]
            pop_f = pop_f[idx]

            # Update mean
            new_x = np.sum(weights[:, None] * pop[:mu], axis=0)
            new_x = np.clip(new_x, -5, 5)

            # Update covariance matrix
            y = (new_x - x) / sigma
            hsig = 1.0 if np.sqrt(np.sum(y ** 2) / (1 - np.exp(-2. * calls / (n * cc)))) < (1 + 2. / n) * np.sqrt(1 - np.exp(-2. * calls / (n * cs))) else 0.0

            pc = (1 - cc) * pc + cc * hsig * y
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y

            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(weights[i] * np.outer(pop[i] - x, pop[i] - x) / sigma ** 2 for i in range(mu))

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - np.exp(-2. * (calls + 1) / (n * cs))) - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)

            x = new_x

            # Periodic eigenvalue clipping for stability
            if calls % 200 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-6, 1e6)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Polish phase: Nelder-Mead around best_x
        if calls < budget and time.time() < deadline:
            from scipy.optimize import minimize
            try:
                remaining = budget - calls
                result = minimize(func, best_x, method='Nelder-Mead',
                                  options={'maxiter': remaining, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
