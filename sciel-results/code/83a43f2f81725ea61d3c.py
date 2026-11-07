"""Black-box continuous optimizer on BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Diversity-first CMA-ES with restart polish, under a hard call + time budget."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 300.0          # hard wall-clock guard (evaluator kills at 330s)
        D = self.dim
        budget = self.budget
        calls = 0

        def f(x):
            nonlocal calls
            calls += 1
            return float(func(np.asarray(x, dtype=float)))

        # ---- Phase 1: diversity-first CMA-ES (spread across 21 peaks) ----
        x, fx = self._cma(
            func=f, D=D, budget=budget,
            x0=np.array([0.0] * D), sigma0=3.0,
            popmax=24, deadline=deadline, calls_ref=[calls],
            explore=True,
        )

        # ---- Phase 2: exploit/polish from the best point ----
        remaining = budget - calls
        if remaining > 0 and time.time() < deadline - 1.0:
            x2, fx2 = self._cma(
                func=f, D=D, budget=remaining,
                x0=x.copy(), sigma0=0.8,
                popmax=12, deadline=deadline, calls_ref=[calls],
                explore=False,
            )
            if fx2 < fx:
                x, fx = x2, fx2

        return x, fx

    def _cma(self, func, D, budget, x0, sigma0, popmax, deadline, calls_ref, explore):
        """Compact CMA-ES. calls_ref is a list holding the global call counter."""
        calls_ref[0] = 0  # local count for this phase; but we track globally via func wrapper
        # NOTE: func() increments the *global* counter (calls) via closure.
        # We use budget to cap the number of evaluations in this phase.

        n = D
        x = np.array(x0, dtype=float)
        sigma = float(sigma0)

        # CMA-ES weights / parameters
        mu = max(1, popmax // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        best_x = x.copy()
        best_f = func(x)
        calls_made = 1
        budget_left = budget - 1

        # exploration: start with a large sigma to scatter; exploitation: smaller
        if explore:
            sigma = max(sigma, 2.5)

        popsize = min(popmax, max(4, int(4 + 3 * np.log(n))))

        while budget_left > 0:
            if time.time() > deadline - 2.0:
                break
            # generate popsize candidates
            for _ in range(popsize):
                if budget_left <= 0 or time.time() > deadline - 2.0:
                    break
                z = np.random.randn(n)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                budget_left -= 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                # (we collect all in a list for the CMA update below)
                if not hasattr(self, "_cma_buf"):
                    self._cma_buf = []
                self._cma_buf.append((y, fy, z))

            # CMA update using the collected candidates
            buf = getattr(self, "_cma_buf", [])
            if len(buf) >= 2:
                buf = sorted(buf, key=lambda t: t[1])
                Y = np.array([b[0] for b in buf[:mu]])
                F = np.array([b[1] for b in buf[:mu]])
                m = Y @ w  # weighted mean (w length == mu)
                # update ps, pc
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (sigma * (m - x) / (sigma + 1e-16) if False else (m - x) / (sigma + 1e-16)))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, int(1.0)))) < (1.4 + 2 / (n + 1)) * chi_n)
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / (sigma + 1e-16)
                # update C
                delta_h = (1 - hsig) * c_c * (2 - c_c)
                C = ((1 + c1 + cmu - c1 - cmu) * C
                     + c1 * (pc @ pc - C)
                     + cmu * np.sum(w * np.outer(Y - x, Y - x), axis=0))
                # symmetrize
                C = 0.5 * (C + C.T)
                # recompute square roots via eigendecomposition
                try:
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.maximum(evals, 1e-12)
                    C_sqrt = (evecs * np.sqrt(evals)) @ evecs.T
                    C_inv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
                    C_inv = (evecs * (1.0 / evals)) @ evecs.T
                except np.linalg.LinAlgError:
                    pass
                # adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = float(np.clip(sigma, 1e-6, 20.0))
                # move mean
                x = m
            # clear buffer
            if hasattr(self, "_cma_buf"):
                del self._cma_buf

            if explore:
                # keep some exploration pressure: do not shrink sigma too fast
                sigma = max(sigma, 0.5)
            else:
                sigma = min(sigma, 2.0)

            if sigma < 1e-8:
                break

        return best_x, best_f


# EVOLVE-BLOCK-END
