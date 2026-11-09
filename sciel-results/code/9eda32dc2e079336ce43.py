"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid (1+1)-ES with diversification and optional CMA-ES polish.
    
    Strategy:
    1. Initial Diversification: Sample diverse points to find a promising basin.
    2. Aggressive (1+1)-ES: Minimal overhead per iteration, rapid step-size adaptation,
       frequent updates - ideal for AOCC which rewards early progress.
    3. Restarts: When stalled, restart from perturbed best or new random point.
    4. Optional CMA-ES polish: If budget remains, use CMA-ES to refine the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)

    def _run_1plus1_es(self, func, x0, fx0, budget, rng, deadline, sigma0=2.0):
        """Run (1+1)-ES with restarts on stall."""
        n = self.dim
        x = np.array(x0, dtype=float)
        fx = fx0
        best_x = x.copy()
        best_f = fx
        calls = 1
        sigma = sigma0

        sigma_inc = 1.3
        sigma_dec = 0.85
        sigma_min = 1e-4
        sigma_max = 10.0

        no_improve = 0
        stall_threshold = 25
        restarts = 0
        max_restarts = 8

        while calls < budget:
            if time.time() > deadline:
                break

            noise = rng.standard_normal(n)
            y = x + sigma * noise
            y = self._clip(y)

            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= sigma_inc
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    no_improve = 0
            else:
                sigma *= sigma_dec
                no_improve += 1

            sigma = max(sigma_min, min(sigma, sigma_max))

            if no_improve > stall_threshold and restarts < max_restarts:
                restarts += 1
                no_improve = 0

                if restarts <= 3:
                    x = best_x + 1.5 * rng.standard_normal(n)
                elif restarts <= 5:
                    x = best_x + 3.0 * rng.standard_normal(n)
                else:
                    x = rng.uniform(-5, 5, n)
                x = self._clip(x)

                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                sigma = 2.0

        return best_x, best_f, calls

    def _run_cma_polish(self, func, x0, budget, rng, deadline, sigma0=0.5):
        """Lightweight CMA-ES for final polishing."""
        n = self.dim
        lam = max(4, 2 * n)
        c = 1.0 / (n + 4)
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_w = np.sum(w[:mu])
        c_sigma = (mu_w + 2) / (n + mu_w / 2)
        c_cov = (2 + mu_w / n) / (n + 2 + mu_w / n)

        C = np.eye(n)
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

        for _ in range(budget):
            if time.time() > deadline or calls >= budget:
                break

            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = self._clip(y)
                offspring.append(y)

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

            idx = np.argsort(fvals)
            fvals_sorted = np.array(fvals)[idx]
            ys_sorted = [offspring[i] for i in idx]

            x_new = np.sum([w[i] * ys_sorted[i] for i in range(mu)], axis=0)
            x_new = self._clip(x_new)

            d_mean = (x_new - x) / sigma
            pc = (1 - c_sigma) * pc + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * d_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * np.linalg.inv(B @ np.diag(D)) @ d_mean

            delta_hs = (np.linalg.norm(ps) > (1.4 + 1 / n) * chi_n)
            h1 = (1 - delta_hs)
            C = (1 - c_cov) * C + c_cov * (
                np.outer(pc, pc) * h1 +
                np.sum([w[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu)]) / sigma**2
            )
            C = (C + C.T) / 2

            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            D = np.sqrt(eigenvalues)
            B = eigenvectors

            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            x = x_new

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Initial Diversification (~15 samples)
        n_init = 15
        init_candidates = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.ones(n) * 1.0,
            np.ones(n) * -1.0,
            np.ones(n) * 3.0,
            np.ones(n) * -3.0,
        ]
        for _ in range(n_init - len(init_candidates)):
            init_candidates.append(rng.uniform(-5, 5, n))

        for x in init_candidates:
            if total_calls >= budget or time.time() > deadline:
                break
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            total_calls = 1

        # Phase 2: (1+1)-ES with restarts - main search
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline:
            x_best, f_best, calls_used = self._run_1plus1_es(
                func, best_x, best_f, remaining, rng, deadline, sigma0=2.0
            )
            total_calls += calls_used
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        # Phase 3: Optional CMA-ES polish with remaining budget
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline:
            x_polish, f_polish, calls_polish = self._run_cma_polish(
                func, best_x, remaining, rng, deadline, sigma0=0.5
            )
            total_calls += calls_polish
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish

        if best_x is None:
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
