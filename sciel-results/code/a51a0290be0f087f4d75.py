"""CMA-ES + L-BFGS-B optimizer for BBOB f12 (bent cigar)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES (global, anisotropic adaptation) + L-BFGS-B (local refinement),
    with an explicit wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        DEADLINE = 295.0          # safety margin under the 300 s wall-clock kill
        dim = self.dim
        budget = self.budget

        calls = [0]

        def f(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        rng = np.random.default_rng(0)

        # ---- Phase 0: cheap multi-start to lock in a good early value (AOCC) ----
        x_best = None
        f_best = np.inf
        # A handful of cheap samples to get a foothold quickly.
        for _ in range(8):
            x = rng.uniform(-5.0, 5.0, dim)
            fx = f(x)
            if fx < f_best:
                f_best, x_best = fx, x.copy()

        # ---- Phase 1: CMA-ES (hand-rolled, anisotropic, no full covariance cost) ----
        # CMA-ES on the bent cigar: adapt sigma and a correlation matrix
        # built from the history of accepted steps.
        n = dim
        # Initial sigma scaled to the box.
        sigma = 2.0
        # Population size (CMA-ES default-ish)
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        # Weights (logarithmic)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # Step-size adaptation
        c_sigma = (mueff + 2) / ((n + 1.3) ** 2 + mueff)
        d_sigma = 1.0 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        # Correlation (C) adaptation
        c_c = (4 + mueff / n) / ((n + 4) ** 2 + mueff / n)
        d_c = 1.0 + 2 * max(0, (mueff - 1.7) / (n + 1.7)) + c_c
        # We maintain a full C (n x n) for the bent-cigar anisotropy.
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        csqrt = 1.0  # sqrt of det(C)

        # chi_n: expected norm of a standard normal vector
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def cma_step():
            nonlocal sigma, C, pc, ps, csqrt, x_best, f_best
            # Generate population
            xs = np.empty((lam, n))
            for i in range(lam):
                # sample from N(0, C)
                z = rng.standard_normal(n)
                xs[i] = x_best + sigma * (C @ z)
            # Evaluate
            fs = np.array([f(xs[i]) for i in range(lam)])
            order = np.argsort(fs)
            # Update best
            if fs[order[0]] < f_best:
                f_best, x_best = fs[order[0]], xs[order[0]].copy()
            # Mean update
            x_mean = np.average(xs[order[:mu]], axis=0, weights=w)
            # ps update
            inv_cs = 1.0 / np.sqrt(1 - (1 - c_c) ** 2)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * inv_cs * (C @ (x_mean - x_best) / sigma)
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) / chi_n) < (1.4 + 2 / (n + 1))
            pc = (1 - c_c) * pc + hs * np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_best) / sigma
            # C update
            delta_h = (1 - hs) * (1 - c_c)
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((n, n))
            for i in range(mu):
                y = (xs[order[i]] - x_best) / sigma
                rankmu += w[i] * np.outer(y, y)
            C = (1 + c_c * delta_h * n - c_c - c_sigma) * C + c_c * rank1 + c_sigma * rankmu
            # Keep C symmetric positive definite-ish
            C = (C + C.T) / 2
            # sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 50.0)
            x_best = x_mean
            # Recompute csqrt occasionally (cheap enough for n=5)
            eigvals = np.linalg.eigvalsh(C)
            det = np.prod(np.clip(eigvals, 1e-12, None))
            csqrt = np.sqrt(det)

        # Run CMA-ES until we have used most of the budget or hit the deadline.
        cma_budget = int(budget * 0.85)
        while calls[0] < cma_budget:
            if time.time() > DEADLINE - 15.0:
                break
            cma_step()

        # ---- Phase 2: L-BFGS-B local refinement from the best point ----
        if time.time() < DEADLINE - 5.0 and calls[0] < budget - 1:
            try:
                res = minimize(
                    f,
                    x_best,
                    method="L-BFGS-B",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 40, "maxfun": max(1, budget - calls[0] - 1),
                             "gtol": 1e-8},
                )
                if res.fun < f_best:
                    f_best, x_best = float(res.fun), np.asarray(res.x, dtype=float)
            except Exception:
                pass

        # Safety: never exceed the budget.
        return x_best, f_best
# EVOLVE-BLOCK-END
