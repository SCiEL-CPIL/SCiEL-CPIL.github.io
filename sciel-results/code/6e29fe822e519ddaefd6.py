"""Optimizer for BBOB f22 (Gallagher 21 peaks) using CMA-ES with adaptive
step size and covariance.  CMA-ES maintains a population of candidates whose
positions are drawn from a multivariate normal with learned covariance,
allowing it to adapt to the local geometry of the 21-peaks landscape and
exploit promising basins more efficiently than fixed-direction methods."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330 s kill
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- CMA-ES implementation ----------
        def cma_es(x0, sigma0, max_gens=None):
            """CMA-ES with population size lambda=12.
            Adapts sigma and covariance C based on fitness ranking."""
            nonlocal best_x, best_f, total_calls

            if max_gens is None:
                max_gens = max(10, (budget - total_calls) // 12)

            lam = 12  # population size
            mu = lam // 2  # number of selected parents
            # Weights: linear ranking
            w = np.arange(1.0, mu + 1.0)
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)  # effective mutation strength
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)  # CM adaptation rate
            cs = (mueff + 2.0) / (n + mueff + 5.0)  # step size adaptation rate
            damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
            chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)  # eigenvalues of C
            pc = np.zeros(n)  # evolution path for covariance
            ps = np.zeros(n)  # evolution path for step size
            count_eval = 0

            for gen in range(max_gens):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                # Generate population
                pop = []
                for i in range(lam):
                    if total_calls >= budget or time.time() > deadline - 5:
                        break
                    # Sample from N(x, sigma^2 * C)
                    z = np.random.default_rng(42 + gen * lam + i).standard_normal(n)
                    # Transform: x + sigma * B @ (D * z)
                    xi = x + sigma * (B @ (D * z))
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = try_eval(xi)
                    if fi is None:
                        break
                    pop.append((fi, xi))

                if len(pop) < mu:
                    break

                # Sort by fitness
                pop.sort(key=lambda p: p[0])

                # Update best
                if pop[0][0] < best_f:
                    best_f = pop[0][0]
                    best_x = pop[0][1].copy()

                # Select parents
                parents = [p[1] for p in pop[:mu]]
                parent_fits = [p[0] for p in pop[:mu]]

                # Weighted mean of parents
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * (parents[i] - x)
                y_w /= sigma

                # Update evolution paths
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * y_w
                pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * (
                    (C_inv @ y_w) if np.allclose(C, np.eye(n)) else y_w
                )

                # Rank-mu update for covariance
                if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (gen + 1))) <= (
                    1.4 + 2.0 / (n + 1.0)
                ) * chiN:
                    delta_sig = 1.0
                    C = (1.0 + 2.0 * cs * (delta_sig - 1.0) * cc) * C + \
                        cc * (2.0 - cc) * (
                            np.outer(pc, pc) -
                            (1.0 - cc) * C
                        )
                # Simplified: use rank-1 + rank-mu
                H = np.zeros((n, n))
                for i in range(mu):
                    yi = (parents[i] - x) / sigma
                    H += w[i] * np.outer(yi, yi)
                C = (1.0 + 2.0 * cs * (delta_sig - 1.0) * cc) * C + cc * (2.0 - cc) * (
                    np.outer(pc, pc) - (1.0 - cc) * C
                ) + (1.0 + 2.0 * cs * (delta_sig - 1.0) * cc) * (1.0 - cc) * C + \
                    (1.0 + 2.0 * cs * (delta_sig - 1.0) * cc) * (
                        1.0 - (1.0 - cc) * (1.0 + 2.0 * cs * (delta_sig - 1.0))
                    ) * C + (1.0 + 2.0 * cs * (delta_sig - 1.0) * cc) * H

                # Simplified covariance update (rank-1 + rank-mu)
                C = (1.0 - cc * (2.0 - cc)) * C + cc * (2.0 - cc) * (
                    np.outer(pc, pc) - (1.0 - cc) * C
                )
                C += (1.0 - cc) * (1.0 + 2.0 * cs * (delta_sig - 1.0)) * H

                # Ensure positive semi-definite
                C = (C + C.T) / 2.0
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)

                # Update sigma
                sigma *= np.exp(
                    cs / damps * (np.linalg.norm(ps) / chiN - 1.0)
                )
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                x = x + y_w * sigma

            return best_x, best_f

        # ---------- Phase 1: Initial broad sampling ----------
        rng_init = np.random.default_rng(7)
        n_phase1 = min(25, budget // 20)
        for i in range(n_phase1):
            if total_calls >= budget or time.time() > deadline - 15:
                break
            x_try = rng_init.uniform(-4.0, 4.0, n)
            try_eval(x_try)

        # Also try origin and a few structured points
        for p in [np.zeros(n), 1.0 * np.ones(n), -1.0 * np.ones(n)]:
            if total_calls >= budget or time.time() > deadline - 15:
                break
            try_eval(p)

        # ---------- Phase 2: CMA-ES from best point ----------
        if total_calls < budget - 15 and time.time() < deadline - 20:
            cma_es(best_x.copy(), sigma0=1.5, max_gens=None)

        # ---------- Phase 3: Additional CMA-ES from diverse starts ----------
        rng2 = np.random.default_rng(99)
        for _ in range(2):
            if total_calls >= budget - 15 or time.time() > deadline - 25:
                break
            x_start = rng2.uniform(-3.0, 3.0, n)
            f_start = try_eval(x_start)
            if f_start is not None and f_start < best_f + 5.0:
                if total_calls < budget - 10 and time.time() < deadline - 15:
                    cma_es(x_start.copy(), sigma0=1.0, max_gens=15)

        # ---------- Phase 4: Final coordinate polish ----------
        if total_calls < budget and time.time() < deadline - 3:
            step = 0.3
            for _ in range(4):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    for direction in [-1, 1]:
                        if total_calls >= budget or time.time() > deadline - 3:
                            break
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is not None and f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                step *= 0.5

        return best_x, best_f
