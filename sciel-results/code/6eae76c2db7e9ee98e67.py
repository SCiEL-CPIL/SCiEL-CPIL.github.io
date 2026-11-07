"""Black-box continuous optimizer for BBOB f6 (Attractive sector).
Global random exploration followed by CMA-ES local exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global random search + CMA-ES (with adaptive step-size control)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim
        start = time.time()
        deadline = start + 285.0          # safety margin before the 300 s kill
        rng = np.random.default_rng(0)

        def time_left():
            return deadline - time.time()

        # ---- Phase 1: global random exploration to find a good start ----
        n_explore = min(50, budget)
        x_best = None
        f_best = np.inf
        for _ in range(n_explore):
            if time_left() < 5.0:
                break
            x = rng.uniform(-5.0, 5.0, n)
            fx = func(x)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()

        if x_best is None:
            x_best = np.zeros(n)
            f_best = func(x_best)

        # ---- Phase 2: CMA-ES local exploitation ----
        x = x_best.copy()
        f = f_best
        # initial step size: explore a fraction of the box around the best point
        sigma = 1.5
        popsize = max(4, int(4 + 3 * np.log(n)))
        # CMA-ES parameters
        c_sigma = (n + 2) / (n + 10 + popsize)
        d_sigma = 1 + 2 * max(0, np.sqrt((popsize - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + n / popsize) / (n + 4 + 2 * popsize / n)
        c_1 = (2 + 2 / (n + 1.3)) / ((n + 2) * (popsize / 2 + 2))
        c_mu = (2 + np.log(popsize / 2 + 2) - 2 / (n + 1.3)) / ((n + 2) * (popsize / 2 + 2))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        evals = 0

        def _inv_sqrtm(M):
            # M is symmetric positive definite; return matrix S with S @ S.T = inv(M)
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 1e-12, None)
            inv_sqrt = 1.0 / np.sqrt(w)
            return V @ np.diag(inv_sqrt) @ V.T

        def _chol(M):
            # return upper-triangular B with B.T @ B = M (M SPD)
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 1e-12, None)
            B = V @ np.diag(np.sqrt(w))
            return B

        while evals < budget and time_left() > 5.0:
            # sample popsize points
            B = _chol(C)
            ys = rng.standard_normal((popsize, n))
            pops = np.array([x + sigma * (y @ B) for y in ys])
            pops = np.clip(pops, -5.0, 5.0)

            fs = np.empty(popsize)
            for i in range(popsize):
                if time_left() < 5.0:
                    # fill the rest with a large value so sorting still works
                    fs[i:] = np.inf
                    break
                fs[i] = func(pops[i])
                evals += 1

            order = np.argsort(fs, kind="stable")
            lam = popsize // 2
            w = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            y_w = np.sum(w[:, None] * (pops[order[:lam]] - x), axis=0)
            y_inv = (pops[order[:lam]] - x) / sigma

            # update evolution path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_w)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (evals + 1)) / chi_n) < 1.4 + 2 / (n + 1)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c)) * y_w
            C = ((1 + 2 * c_1 * hsig * (2 - c_c) - c_1 - c_mu) * C
                 + c_1 * np.outer(pc, pc)
                 + c_mu * np.sum(w[:, None, None] * np.outer(y_inv, y_inv), axis=0))
            # re-symmetrize
            C = 0.5 * (C + C.T)

            # update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 10.0)
            sigma = max(sigma, 1e-4)

            # move to best
            x_new = pops[order[0]].copy()
            f_new = fs[order[0]]
            if f_new < f:
                f = f_new
                x = x_new
            else:
                # keep the mean update anyway for distribution; but track best separately
                x = x_new
                # restore best point reference
                x = pops[order[0]]

        return x, f
# EVOLVE-BLOCK-END
