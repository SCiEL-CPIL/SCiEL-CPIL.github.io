"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, x0, sigma0, func, rng, max_calls, deadline):
        """Run CMA-ES from x0 with initial step sigma0."""
        n = self.dim
        x = x0.copy()
        sigma = sigma0
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        lam = 4 * n  # population size
        mu = lam // 2
        # Weights
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = int(1.0 / np.sum(weights ** 2))
        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Initial state
        ps = np.zeros(n)
        pc = np.zeros(n)
        C = np.eye(n)
        invC = np.eye(n)
        sqrtC = np.eye(n)
        invsqrtC = np.eye(n)
        dnorm = np.sqrt(n) + 0.5 * np.sqrt(n) - 1

        def _update_chol(C, invC, sqrtC, invsqrtC):
            """Update Cholesky factors."""
            L = np.linalg.cholesky(C)
            sqrtC = L
            invsqrtC = np.linalg.inv(L)
            return L, invsqrtC

        # Initial Cholesky
        sqrtC, invsqrtC = _update_chol(C, invC, sqrtC, invsqrtC)

        for _ in range(max_calls - 1):
            if calls >= max_calls:
                break
            if time.time() > deadline:
                break

            # Sample population
            Z = rng.standard_normal((lam, n))
            Y = np.array([x + sigma * (invsqrtC @ z) for z in Z])
            Y = np.clip(Y, -5, 5)

            # Evaluate
            F = np.array([func(y) for y in Y])
            calls += lam

            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Mean update
            yw = np.zeros(n)
            for i in range(mu):
                yw += weights[i] * (Y[i] - x) / sigma
            x_new = x + yw

            # Step-size control (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ yw)
            chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1)) > (1.4 + 2 / (n + 1)) * chiN:
                sigma *= np.exp(cs * (np.linalg.norm(ps) / chiN - 1))

            # Covariance matrix adaptation
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** calls) < (1.4 + 2 / (n + 1)) * chiN
            hsig = float(hsig)
            pc = (1 - cc) * pc + hsig * cc * yw
            rank_one = np.outer(pc, pc)
            C = (1 + c1 * hsig * (cc * (2 - cc)) - c1) * C + c1 * rank_one

            # Rank-mu update
            if mueff > 1:
                diff = (Y[:mu] - x) / sigma
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(diff[i], diff[i])
                C = (1 + c1 * hsig * (cc * (2 - cc) - c1) - cmu) * C + cmu * rank_mu

            # Ensure positive definiteness
            C = (C + C.T) / 2
            try:
                sqrtC, invsqrtC = _update_chol(C, invC, sqrtC, invsqrtC)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-10 * np.eye(n)
                try:
                    sqrtC, invsqrtC = _update_chol(C, invC, sqrtC, invsqrtC)
                except np.linalg.LinAlgError:
                    break

            x = x_new
            fx = F[0]

        return x, fx, calls

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 295  # Safety margin before 330s

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: CMA-ES from multiple diverse starting points
        n_starts = 3
        calls_per_start = max(1, (self.budget - 10) // n_starts)

        for i in range(n_starts):
            if total_calls >= self.budget - 5:
                break
            if time.time() > deadline:
                break

            if i == 0:
                x0 = np.zeros(self.dim)
                sigma0 = 2.0
            elif i == 1:
                x0 = rng.uniform(-4, 4, self.dim)
                sigma0 = 3.0
            else:
                x0 = rng.uniform(-5, 5, self.dim)
                sigma0 = 4.0

            remaining = self.budget - total_calls - 5
            max_calls = min(calls_per_start, remaining)
            if max_calls < 5:
                break

            x, fx, used = self._cmaes(x0, sigma0, func, rng, max_calls, deadline)
            total_calls += used

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local polish with (1+1)-ES from best point
        if best_x is not None and total_calls < self.budget - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            polish_budget = self.budget - total_calls - 5

            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Ensure we made at least one call
        if best_x is None:
            x0 = np.zeros(self.dim)
            best_f = func(x0)
            best_x = x0

        return best_x, best_f
# EVOLVE-BLOCK-END
