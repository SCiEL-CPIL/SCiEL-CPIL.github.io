# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES for box-constrained continuous minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 270.0  # safety margin under the 300 s kill
        dim = self.dim
        budget = self.budget

        # Global best
        best_x = None
        best_f = np.inf

        def is_deadline():
            return (time.perf_counter() - t0) > deadline

        def cmaes(x0, f0, sigma0, remaining, rng):
            """Run CMA-ES starting from (x0, f0). Returns (x_best, f_best)."""
            n = dim
            m = max(4, int(round((4 + int(np.log(n))) * 1)))  # population size
            m = min(m, max(4, remaining // 4))
            if m < 4:
                m = 4

            x = x0.copy()
            sigma = sigma0
            fx = f0

            # Covariance / eigendecomposition state
            C = np.eye(n)
            # Initialize eigen decomposition
            try:
                D, B = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                D, B = np.ones(n), np.eye(n)
            D = np.maximum(D, 1e-12)
            sqrtC = B @ np.diag(np.sqrt(D)) @ B.T
            inv_sqrtC = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

            # Step sizes
            cm = 1.0 / (n + 3 + 2.0 / (n + 1))
            cs = (n + 1.7 - n / 2) / (n + n / 2 + 2)
            cm_1 = 1.0 - cm
            cs_1 = 1.0 - cs
            damps = 1.0 + 2.0 * max(0.0, (np.sqrt((n + 1) - 1) - 1) / (n + 3) - 1.0) + cs
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
            cc = (4.0 + n / n) / (4.0 + 4.0 * n)  # = 1
            cc_1 = 1.0 - cc

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            calls_used = 1  # f0 already counted
            best_x_local = x.copy()
            best_f_local = fx

            max_iter = max(1, remaining // m)
            for _ in range(max_iter):
                if is_deadline():
                    break
                if calls_used >= remaining:
                    break

                # Generate population
                ys = rng.standard_normal((m, n))
                # Normalize using current sqrtC
                ys = ys @ inv_sqrtC.T
                pops = np.tile(x, (m, 1)) + sigma * ys
                # Clip to box
                np.clip(pops, -5.0, 5.0, out=pops)

                # Evaluate
                fs = np.empty(m)
                for i in range(m):
                    fs[i] = func(pops[i])
                    calls_used += 1
                    if fs[i] < best_f_local:
                        best_f_local = fs[i]
                        best_x_local = pops[i].copy()
                    if is_deadline():
                        break
                if is_deadline():
                    break

                # Sort
                order = np.argsort(fs)
                mu = m // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mu_eff = 1.0 / (w @ w)
                x_mean = (w[:mu, None] * pops[order[:mu]]).sum(axis=0)

                # Update p_sigma
                p_sigma = cs_1 * p_sigma + np.sqrt(cs * (2.0 - cs) * mu_eff) / chi_n * (
                    inv_sqrtC @ (x_mean - x) / sigma
                )

                # h_sigma
                p_sigma_norm = np.linalg.norm(p_sigma)
                h_sigma = (
                    p_sigma_norm
                    / np.sqrt(1.0 - (1.0 / (1.0 + 0.5 * iter_count) ** cs))
                    / chi_n
                    < (1.3 + 2.0 / (n + 1))
                )

                # Update p_c
                p_c = cc_1 * p_c + h_sigma * cc * (x_mean - x) / sigma

                # Update C
                C = (
                    cm_1 * C
                    + cm
                    * (
                        np.outer(p_c, p_c)
                        + (h_sigma * (2.0 - h_sigma) * cc * (1.0 - cc_1)) * C
                    )
                    + mu_eff * sigma * sigma * (
                        (w[:mu, None] * (pops[order[:mu]] - x) / sigma)
                        @ ((w[:mu, None] * (pops[order[:mu]] - x) / sigma)).T
                    )
                )
                # Symmetrize
                C = 0.5 * (C + C.T)

                # Update sigma
                sigma *= np.exp(
                    cs / chi_n * (p_sigma_norm / chi_n - 1.0)
                )
                sigma = np.clip(sigma, 1e-12, 1e3)

                # Re-eigendecompose periodically
                if _ % 5 == 0:
                    try:
                        D, B = np.linalg.eigh(C)
                    except np.linalg.LinAlgError:
                        D, B = np.ones(n), np.eye(n)
                    D = np.maximum(D, 1e-12)
                    inv_sqrtC = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

                x = x_mean
                fx = fs[order[0]]
                if fx < best_f_local:
                    best_f_local = fx
                    best_x_local = pops[order[0]].copy()

                iter_count = getattr(self, "_cma_iter", 0) + 1
                self._cma_iter = iter_count

            return best_x_local, best_f_local, calls_used

        # Multi-start CMA-ES
        rng = np.random.default_rng(0)
        self._cma_iter = 0

        # First, a quick random scan to find a decent starting point
        scan_calls = min(20, budget // 10)
        for i in range(scan_calls):
            x_try = rng.uniform(-5.0, 5.0, dim)
            f_try = func(x_try)
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
            if is_deadline():
                break

        # Remaining budget for CMA-ES restarts
        remaining = budget - scan_calls
        # Allocate restarts: first restart gets most of the budget
        n_restarts = 3
        # Budget allocation: 60%, 25%, 15%
        allocs = [int(remaining * 0.6), int(remaining * 0.25), int(remaining * 0.15)]
        # Ensure last gets the rest
        allocs[-1] = remaining - allocs[0] - allocs[1]
        if allocs[-1] < 8:
            allocs[1] = max(8, allocs[1] - (8 - allocs[-1]))
            allocs[-1] = max(8, remaining - allocs[0] - allocs[1])

        for r in range(n_restarts):
            if is_deadline():
                break
            if r == 0:
                x0 = best_x.copy()
                f0 = best_f
            else:
                x0 = rng.uniform(-5.0, 5.0, dim)
                f0 = func(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

            sigma0 = 1.5
            xb, fb, _ = cmaes(x0, f0, sigma0, allocs[r], rng)
            if fb < best_f:
                best_f = fb
                best_x = xb.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
