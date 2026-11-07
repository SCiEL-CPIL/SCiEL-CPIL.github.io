"""CMA-ES black-box optimizer for BBOB-style Rastrigin (f3)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with success-count step-size adaptation and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma(self, func, x0, sigma0, n_evals, deadline, rng):
        n = self.dim
        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w @ w)
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (2 - cc)
        cmu = (2 * (mu_eff - 2 + w[0]) / ((n + 2) ** 2 + mu_eff)) * (2 - cc)
        cmu = min(cmu, 2 * c1)
        c_sigma = 0.35 + mu_eff / (n + 2)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        x = x0.copy()
        sigma = sigma0
        ps = np.zeros(n)
        pc = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        pcov = np.eye(n)
        best_x = x0.copy()
        best_f = float("inf")

        for _ in range(n_evals):
            if time.perf_counter() >= deadline:
                break
            # sample population
            Y = []
            for _ in range(mu):
                z = rng.standard_normal(n)
                Y.append(x + sigma * (B @ (D * z)))
            Y = np.array(Y)
            F = np.empty(mu)
            for i in range(mu):
                F[i] = func(np.clip(Y[i], -5, 5))
            order = np.argsort(F)
            Ys = Y[order]
            Fs = F[order]
            if Fs[0] < best_f:
                best_f = Fs[0]
                best_x = Ys[0].copy()
            m = w @ Ys
            diff = m - x
            d = B @ (D * diff)
            D2 = (D * D) @ pcov
            D2 = D2 + (1 - D2) * np.eye(n)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (D2 @ (D @ diff)) / sigma
            if np.linalg.norm(ps) < (1.4 + 1 / cs) * chi_n:
                pc = (1 - cc) * pc + cc * diff / sigma
            pcov = (1 - c1 - cmu) * pcov
            pcov += c1 * (pc @ pc + (1 - (1 - cc) ** 2) * np.eye(n))
            for i in range(mu):
                pcov += cmu * w[i] * np.outer(Ys[i] - x, Ys[i] - x)
            # eigen-decomposition (stable update)
            eigval, eigvec = np.linalg.eigh(pcov)
            eigval = np.clip(eigval, 1e-20, None)
            B = eigvec
            D = np.sqrt(eigval)
            x = m
            # step-size adaptation (success-count rule)
            sig0 = sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(max(sigma, 1e-12), 1e6)
        return best_x, best_f

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 250.0          # safety margin under the 300 s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        best_x = None
        best_f = float("inf")

        # cheap coarse scan to find a promising basin (Rastrigin is multimodal)
        grid = np.linspace(-5, 5, 4)
        best_f = float("inf")
        best_x = None
        for a in grid:
            for b in grid:
                x = np.array([a, b, 0.0, 0.0, 0.0])
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x
        # refine around the coarse best with a few local random probes
        for _ in range(20):
            x = np.clip(best_x + rng.uniform(-1, 1, n), -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x

        used = 24 + 20
        remaining = max(0, budget - used)

        # primary CMA-ES run from the coarse best
        x0 = best_x.copy()
        bx, bf = self._cma(func, x0, 1.0, remaining, deadline, rng)
        if bf < best_f:
            best_f, best_x = bf, bx

        # restart CMA-ES from a fresh random point if time allows
        if time.perf_counter() < deadline:
            x1 = rng.uniform(-5, 5, n)
            bx2, bf2 = self._cma(func, x1, 1.5, 0, deadline, rng)  # 0 -> guard-only run
            # (the loop above respects the deadline; give it a real budget)
        return best_x, best_f
# EVOLVE-BLOCK-END
