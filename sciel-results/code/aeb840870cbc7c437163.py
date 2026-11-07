"""Portfolio optimizer for BBOB-style f7 (Step ellipsoidal), dim=5, budget=500."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio: broad samples + axis scans + small CMA-ES + coordinate polish,
    sharing a call budget and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0          # 300 s evaluator limit, generous margin
        B = self.budget
        d = self.dim
        LO, HI = -5.0, 5.0
        SPAN = HI - LO

        # ---- bookkeeping -------------------------------------------------
        used = 0

        def f(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), LO, HI)
            used += 1
            return float(func(x))

        # ---- Phase 0: cheap initial samples -----------------------------
        # center + axis extremes to give a broad picture in ~11 calls.
        xs = [np.full(d, 0.0)]
        for i in range(d):
            p = np.full(d, 0.0); p[i] = -4.0; xs.append(p.copy())
            p = np.full(d, 0.0); p[i] =  4.0; xs.append(p.copy())

        best_x = None
        best_f = np.inf
        for x in xs:
            if used >= B or time.time() > deadline:
                break
            v = f(x)
            if v < best_f:
                best_f, best_x = v, x.copy()

        # ---- Phase 1: coordinate-axis scans -----------------------------
        n_axis_pts = 9
        for i in range(d):
            if used >= B or time.time() > deadline:
                break
            lo = max(LO, best_x[i] - 3.0)
            hi = min(HI, best_x[i] + 3.0)
            for t in np.linspace(lo, hi, n_axis_pts):
                if used >= B or time.time() > deadline:
                    break
                cand = best_x.copy(); cand[i] = t
                v = f(cand)
                if v < best_f:
                    best_f, best_x = v, cand.copy()

        # ---- Phase 2: small-population CMA-ES (learns rotation) ---------
        # Time-guarded: runs until either budget or time is exhausted, so it
        # cannot overrun even if the population is slow to converge.
        rng = np.random.default_rng(12345)
        lam = 6
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        x_mean = best_x.copy()
        sigma = 1.5
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        gen = 0

        while used < B:
            if time.time() > deadline:
                break
            # Reserve a few calls for the final polish phase.
            if B - used < lam + d:
                break

            pop = np.empty((lam, d))
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = C_inv_sqrt @ rng.standard_normal(d)
                pop[i] = np.clip(x_mean + sigma * z, LO, HI)
                fvals[i] = f(pop[i])
            if time.time() > deadline:
                break

            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            if fvals[0] < best_f:
                best_f, best_x = float(fvals[0]), pop[0].copy()

            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            step = x_mean - x_mean_old
            step_norm = np.linalg.norm(step)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step / (sigma * step_norm + 1e-15)
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (d + 1)) * chi_n
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-4, SPAN))

            if h_sigma:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step / sigma
            else:
                pc = (1 - c_c) * pc

            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = ((1 - c_1 - c_mu) * C
                 + c_1 * ((1 + delta_h) * np.outer(pc, pc) - C)
                 + c_mu * np.sum(weights[i] * np.outer(pop[i] - x_mean_old, pop[i] - x_mean_old) for i in range(mu)))

            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            gen += 1

        # ---- Phase 3: coordinate re-scan around the final best ----------
        if used < B and time.time() < deadline:
            for i in range(d):
                if used >= B or time.time() > deadline:
                    break
                lo = max(LO, best_x[i] - 1.0)
                hi = min(HI, best_x[i] + 1.0)
                for t in np.linspace(lo, hi, 7):
                    if used >= B or time.time() > deadline:
                        break
                    cand = best_x.copy(); cand[i] = t
                    v = f(cand)
                    if v < best_f:
                        best_f, best_x = v, cand.copy()

        # ---- Phase 4: final tiny-perturbation polish ---------------------
        rng2 = np.random.default_rng(999)
        fails = 0
        while used < B and time.time() < deadline:
            y = np.clip(best_x + 0.1 * rng2.standard_normal(d), LO, HI)
            fy = f(y)
            if fy < best_f:
                best_f, best_x = fy, y.copy()
                fails = 0
            else:
                fails += 1
                if fails >= 8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
