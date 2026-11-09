"""Rewritten optimizer for BBOB f13 (Sharp Ridge)."""
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) using:
    1. Halton sequence initialization for broad early coverage (AOCC).
    2. CMA-ES (1+λ) with adaptive population for ridge following.
    3. Deadline guard to stay under the time limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, dim, skip=1):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(n):
            idx = i + skip
            for j in range(dim):
                base = j + 2  # bases: 2, 3, 5, 7, 11
                f = 1.0
                res = 0.0
                v = idx
                while v > 0:
                    f /= base
                    res += f * (v % base)
                    v //= base
                points[i, j] = res
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            fx = func(np.clip(x, -5, 5))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Halton initialization - broad coverage for AOCC
        n_init = min(50, budget // 4)
        halton_pts = self._halton(n_init, dim, skip=1)
        init_pts = halton_pts * 10.0 - 5.0  # map to [-5, 5]^dim

        for p in init_pts:
            if time.time() > deadline or calls >= budget:
                break
            evaluate(p)

        if best_x is None:
            return None, np.inf

        # Phase 2: CMA-ES refinement from the best point
        # Use a (1+4) strategy for dim=5
        lam = 4
        pop_size = 1 + lam
        sigma = 2.0  # initial step size
        x_mean = best_x.copy()
        f_mean = best_f

        # CMA-ES state
        C = np.eye(dim)
        alpha_mu = 1.0
        cs = (lam // 2 + 1) / (dim + lam)
        ds = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + cs
        c1 = 2 / ((dim + 1.3) ** 2 + lam)
        cmu = min(0.5 - c1, 2 * ((lam // 2 + 1) - 2 + 1.0 / (lam // 2)) / ((dim + 2) ** 2 + lam))
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Remaining budget for CMA-ES
        remaining = budget - calls

        # Run CMA-ES iterations
        for gen in range(remaining // pop_size + 1):
            if time.time() > deadline or calls >= budget:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                if calls >= budget:
                    break
                B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                z = rng.standard_normal(dim)
                y = x_mean + sigma * B @ z
                fy = evaluate(y)
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            f_best_off = offspring[0][0]
            x_best_off = offspring[0][1]

            # Update mean
            x_mean = x_best_off.copy()
            f_mean = f_best_off

            # Update evolution paths
            # Inverse square root of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            y_plus = C_sqrt_inv @ (x_best_off - x_mean)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * lam) * y_plus
            hsig = 1.0 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls - budget + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs) ** (2 * (calls - budget + 1)))) else 0.0
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * lam) * (x_best_off - x_mean)

            # Update covariance
            delta_sigma = 1 + cmu * (min(5, (np.linalg.norm(pc) ** 2 / (1 - (1 - c1) ** (2 * (calls - budget + 1))) - 1) / (dim + 1)))
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu / delta_sigma * np.sum(
                [np.outer(C_sqrt_inv @ (offspring[i][1] - x_mean), C_sqrt_inv @ (offspring[i][1] - x_mean)) for i in range(lam)], axis=0
            )

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(lam) - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Ensure C is positive definite
            eigvals_c, _ = np.linalg.eigh(C)
            if np.min(eigvals_c) < 1e-12:
                C = C + (1e-12 - np.min(eigvals_c)) * np.eye(dim)

        return best_x, best_f
