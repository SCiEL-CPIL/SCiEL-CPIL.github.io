"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, budget, deadline):
        n = self.dim
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Step size
        sigma = 2.0
        # Weights: log-sum weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w * w).sum()
        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 - mu_eff / (n + 1.8) + c_sigma) / (
            (n + 1.8) * (2 + mu_eff / (n + 1.8))
        )
        c_mu = (
            2 * (mu_eff - 2 + 2.0 / mu_eff)
            / ((n + 2) * (2 - c_1 + mu_eff / (n + 1.8)))
        )
        c_mu = min(c_mu, 2 - c_1)

        # State
        x = x0.copy()
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        Cinv_sqrt = np.eye(n)
        # Cholesky of C
        try:
            L = np.linalg.cholesky(C)
            Linv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            Linv = np.eye(n)

        # Track best
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        # Precompute
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lam, n))
            for i in range(lam):
                z = np.random.randn(n)
                ys[i] = x + sigma * (Linv @ z)

            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                fs[i] = func(ys[i])
                calls += 1

            if calls >= budget:
                break

            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()

            # Mean of top mu
            m_new = (w[:, None] * ys[:mu]).sum(axis=0)

            # Check time again
            if time.time() > deadline:
                break

            # Update evolution paths
            d_old = m_new - x
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * d_old
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - budget))) if calls < budget else 0
            # Simpler hs calculation
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, calls)))
            delta_hs = 1.0 if hs < (1.4 + 1 / n) * chi_n else 0.0

            pc = (1 - c_c) * pc + delta_hs * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * d_old

            # Update covariance
            v = (m_new - x) / sigma
            C = (
                (1 - c_1 - c_mu) * C
                + c_1 * (pc * pc.T + (1 - delta_hs) * c_c * (2 - c_c) * np.eye(n))
                + c_mu * (w[:, None] * (v[:mu] * v[:mu].T)).sum(axis=0)
            )

            x = m_new

            # Adapt sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 50.0)
            sigma = max(sigma, 1e-10)

            # Update Cinv_sqrt via Cholesky
            try:
                L = np.linalg.cholesky(C)
                Linv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Regularize
                C += 1e-8 * np.eye(n)
                try:
                    L = np.linalg.cholesky(C)
                    Linv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    Linv = np.eye(n)

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        budget = self.budget
        n = self.dim

        # Try CMA-ES from a few random starting points if time allows
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: CMA-ES from origin (0 is often good for shifted functions)
        x0 = np.zeros(n)
        # Give most budget to first run
        b1 = min(budget, 400)
        try:
            bx, bf, used = self._cma_es(func, x0, b1, deadline)
            total_calls += used
            if bf < best_f:
                best_f = bf
                best_x = bx
        except Exception:
            pass

        # Phase 2: If time and budget remain, try another start
        if total_calls < budget and time.time() < deadline - 10:
            x1 = np.random.randn(n) * 2
            b2 = min(budget - total_calls, 100)
            if b2 >= 4:
                try:
                    bx, bf, used = self._cma_es(func, x1, b2, deadline)
                    total_calls += used
                    if bf < best_f:
                        best_f = bf
                        best_x = bx
                except Exception:
                    pass

        # Phase 3: Local refinement with (1+1)-ES from best
        if total_calls < budget and time.time() < deadline - 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while total_calls < budget and time.time() < deadline - 2:
                y = np.clip(x + sigma * np.random.randn(n), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                else:
                    sigma *= 0.85
                    if sigma < 1e-8:
                        break

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
