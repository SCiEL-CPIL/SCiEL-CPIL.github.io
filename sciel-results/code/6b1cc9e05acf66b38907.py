"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar


class Optimizer:
    """Global basin (Sobol) + BFGS local refinement + short CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        dim = self.dim
        budget = self.budget
        deadline = t0 + 290.0          # explicit safety margin vs 300s wall-clock

        def timed():
            return time.time() < deadline

        # Deterministic Sobol-like global sample (Halton) for reproducible basin.
        def halton(n, base):
            out = np.zeros(n)
            f, r = 1.0, 0.0
            for i in range(1, n + 1):
                idx = i
                while idx:
                    idx //= base
                    f /= base
                    r += f * (idx % base)
                out[i - 1] = r
            return out

        def sobol_grid(count):
            pts = np.empty((count, dim))
            for d in range(dim):
                # small primes for Halton dimensions
                base = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)[d % 10]
                pts[:, d] = halton(count, base)
            return pts

        # ---------- Phase 1: global basin ----------
        n_global = min(budget, 120)
        G = sobol_grid(n_global)
        vals = np.empty(n_global)
        for i in range(n_global):
            if not timed():
                break
            vals[i] = func(G[i])
        # keep the best few as multi-start seeds
        order = np.argsort(vals)[:5]
        seeds = G[order].copy()

        # ---------- Phase 2: BFGS local refinement from each seed ----------
        best_x, best_f = seeds[0], vals[order[0]]
        for s in seeds:
            if not timed():
                break
            res = minimize(func, s.copy(), method="BFGS",
                           options={"maxiter": 60, "maxls": 40})
            if res.fun < best_f:
                best_f, best_x = res.fun, res.x

        # ---------- Phase 3: short CMA-ES-style local polish ----------
        # Deterministic CMA-ES with small pop, few generations, strict time guard.
        rng = np.random.default_rng(1234)
        x = best_x.copy()
        f = best_f
        sigma = 0.3
        pop = 10
        while timed():
            gen_budget = 12
            for _ in range(gen_budget):
                if not timed():
                    break
                N = rng.multivariate_normal(np.zeros(dim), np.eye(dim), size=pop)
                cand = np.clip(x + sigma * N, -5, 5)
                cf = np.array([func(cand[i]) for i in range(pop)])
                w = np.sort(cf)
                if w[0] < f:
                    f = w[0]
                    x = cand[np.argmin(cf)]
                # 1/5-ish adaptation
                if np.mean(cf < f + 1e-12) > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                if sigma < 1e-4:
                    break
            if not timed():
                break
            if sigma < 1e-4:
                break
            # small deterministic restarts of sigma to escape local flatness
            if not timed():
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
