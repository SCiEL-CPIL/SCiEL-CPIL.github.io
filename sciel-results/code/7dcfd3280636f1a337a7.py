"""Black-box continuous optimizer on BBOB-style f3 (Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Rastrigin-aware restarts, deadline guard, and adaptive strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _evaluate(self, x, func, calls_ref, deadline):
        """Evaluate a single point, respecting budget and deadline."""
        if calls_ref[0] >= self.budget:
            return None
        if time.time() > deadline:
            return None
        x = np.clip(x, -5, 5)
        fx = float(func(x))
        calls_ref[0] += 1
        return fx

    def _cma_es(self, x0, sigma0, func, calls_ref, deadline, max_steps, rng):
        """
        Run CMA-ES from x0 with initial sigma0.
        Returns (best_x, best_f) found during this run.
        """
        n = self.dim
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)

        # CMA-ES parameters
        lam = 2 * (int(4 + 3 * np.log(n)))  # population size ~ 2*(4+3*ln5) ≈ 2*9=18
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cc) / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)
        count_evals = 0
        stagnation = 0

        best_x = x.copy()
        best_f = np.inf

        for step in range(max_steps):
            if time.time() > deadline or calls_ref[0] >= self.budget:
                break

            # Generate population
            Y = np.zeros((lam, n))
            F = np.full(lam, np.inf)
            for i in range(lam):
                if time.time() > deadline or calls_ref[0] >= self.budget:
                    break
                z = rng.standard_normal(n)
                # Cholesky of C for sampling
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                Y[i] = x + sigma * (L @ z)
                fx = self._evaluate(Y[i], func, calls_ref, deadline)
                if fx is not None:
                    F[i] = fx
                    count_evals += 1

            # Sort by fitness
            valid = np.where(np.isfinite(F))[0]
            if len(valid) < 2:
                break
            idx = valid[np.argsort(F[valid])]
            Ys = Y[idx[:mu]]
            Fs = F[idx[:mu]]

            # Update best
            if Fs[0] < best_f:
                best_f = Fs[0]
                best_x = Ys[0].copy()
                stagnation = 0
            else:
                stagnation += 1

            # Update mean
            x_new = np.sum(weights[:, None] * Ys, axis=0)

            # Update covariance matrix
            y = (x_new - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (count_evals / (cs * n)))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - cc) * pc + hsig * cc * y
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(Ys[i] - x, Ys[i] - x)
            C = (1 + c1 * cc * (np.trace(C) / n - 1) + c1 * (cc * (2 - cc) - 1) - cmu * np.sum(weights)) * C \
                + c1 * (rank_one + cc * (cc * (2 - cc) - 1) * (np.trace(C) / n - 1) * np.eye(n)) \
                + cmu * rank_mu / (sigma ** 2)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-4, 10.0)

            x = x_new
            count_evals += 1

            # Stagnation detection
            if stagnation > 20 and step > 10:
                break

        return best_x, best_f

    def __call__(self, func):
        n = self.dim
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_ref = [0]

        # Phase 1: Broad exploration
        for i in range(min(15, self.budget // 10)):
            if time.time() > deadline or calls_ref[0] >= self.budget:
                break
            x = rng.uniform(-5, 5, n)
            fx = self._evaluate(x, func, calls_ref, deadline)
            if fx is not None and fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from multiple starting points
        n_restarts = 6
        budget_per_restart = max(40, (self.budget - calls_ref[0] - 30) // n_restarts)

        start_points = []
        if best_x is not None:
            start_points.append(best_x.copy())
        start_points.append(np.zeros(n))
        start_points.append(rng.uniform(-4, 4, n))
        start_points.append(np.ones(n) * 3.0)
        start_points.append(-np.ones(n) * 3.0)
        start_points.append(rng.uniform(-5, 5, n))
        start_points.append(np.array([4.0, -4.0, 4.0, -4.0, 4.0]))
        start_points.append(rng.uniform(-2, 2, n))

        rng.shuffle(start_points)

        for sp in start_points:
            if time.time() > deadline or calls_ref[0] >= self.budget:
                break

            fx = self._evaluate(sp, func, calls_ref, deadline)
            if fx is not None and fx < best_f:
                best_f = fx
                best_x = sp.copy()

            sigma = 2.0
            bx, bf = self._cma_es(sp, sigma, func, calls_ref, deadline,
                                  budget_per_restart, rng)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

            if best_f < 1.0:
                break

        # Phase 3: Fine-tuning
        if best_x is not None and time.time() < deadline and calls_ref[0] < self.budget:
            remaining = self.budget - calls_ref[0]
            if remaining > 30:
                bx, bf = self._cma_es(best_x, 0.5, func, calls_ref, deadline,
                                      min(remaining, 60), rng)
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = float('inf')

        return best_x, best_f
# EVOLVE-BLOCK-END
