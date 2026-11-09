"""Optimizer for BBOB-style f6 (Attractive sector), dim=5, box [-5,5]^5."""
import time
import numpy as np


class Optimizer:
    """Population CMA-ES with 1/5 success-rule sigma adaptation + NM polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # --- Initial population sampling ---
        # A few spread-out seeds to escape the "attractive sector" trap.
        lam = 8  # population size
        mu = max(1, lam // 2)
        x0 = np.zeros(dim)
        sigma = 1.5
        C = np.eye(dim)
        B = np.eye(dim)
        D = 1.0

        # candidate initial means
        seeds = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
        ]
        # evaluate seeds to pick a good starting mean
        best_seed, best_seed_f = None, np.inf
        for s in seeds:
            fs = func(np.clip(s, -5, 5))
            if fs < best_seed_f:
                best_seed_f = fs
                best_seed = s
        x = best_seed.astype(float).copy()
        fx = best_seed_f
        calls = len(seeds)

        # cumulative weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        cs = (c_sigma + 1) * (1 + c_sigma)
        hs = (1 + 0.3 * (1 - c_sigma)) / (1 + 0.3 * c_sigma)
        one_over_sqrt_detD = 1.0
        one_over_sqrtD = np.ones(dim)
        one_over_sqrtDinv = np.ones(dim)

        def cholesky(C):
            L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            return L

        L = cholesky(C)

        # --- Main CMA-ES loop ---
        while calls < budget:
            # time guard with safety margin
            if time.time() > deadline - 15.0:
                break

            # generate population
            z = rng.standard_normal((lam, dim))
            pop = np.array([x + sigma * (L @ z[i].T).T for i in range(lam)])
            pop = np.clip(pop, -5, 5)

            fvals = np.empty(lam)
            for i in range(lam):
                fvals[i] = func(pop[i])
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline - 15.0:
                    # finish evaluating current pop partially? stop generating new
                    fvals[i + 1:] = np.inf
                    break

            # track best
            if np.min(fvals) < fx:
                idx_best = int(np.argmin(fvals))
                fx = fvals[idx_best]
                x = pop[idx_best].copy()

            if calls >= budget or time.time() > deadline - 15.0:
                break

            # sort by fitness
            order = np.argsort(fvals)
            f_sorted = fvals[order]
            x_sorted = pop[order]

            # weighted mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * x_sorted[i]
            y_w = y_w / np.sum(weights[:mu])

            # update B, D, C from x_sorted
            C_new = np.zeros((dim, dim))
            for i in range(mu):
                d = x_sorted[i] - x
                C_new += weights[i] * np.outer(d, d)
            # rank-one
            ys = (y_w - x) / sigma
            C_new += c_1 * (np.outer(ys, ys) + (1 - c_1) * np.eye(dim))
            C_new += c_mu * np.sum(weights[:mu] ** 2) * np.eye(dim) * 0  # (simplified: use standard form)
            # standard CMA-ES rank-mu with the mean correction
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(ys, ys) + (1 - c_1) * np.eye(dim)) \
                + c_mu * (C_new / (c_mu) * 0)  # keep simple: use rank-one + covariance term
            # Recompute properly:
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(ys, ys) + (1 - c_1) * np.eye(dim))
            C = 0.5 * (C + C.T)

            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sign(sigma / sigma) * ys / chiN
            hs = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls))) < (1.4 + 2 / (dim + 1)) * chiN)
            sigma = sigma * np.exp(cs * (np.linalg.norm(p_sigma) / chiN - 1) * 0.3)
            sigma = np.clip(sigma, 1e-4, 50.0)

            # p_c
            p_c = (1 - c_c) * p_c + hs * c_c * ys
            # update C with rank-one of p_c
            C = (1 - c_1 - c_mu) * C + (c_1 + c_mu * hs * (1 - hs)) * np.outer(p_c, p_c) \
                + c_1 * (np.outer(ys, ys) + (1 - c_1) * np.eye(dim))
            C = 0.5 * (C + C.T)

            # recompute Cholesky and x = weighted mean
            try:
                L = cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = cholesky(C)
                B = np.eye(dim)
                D = 1.0
            x = y_w

        # --- Nelder-Mead polish on the best point ---
        if time.time() < deadline - 5.0 and calls < budget:
            try:
                x_best = np.clip(x, -5, 5)
                f_best = func(x_best)
                calls += 1
                # NM setup
                simplex = [x_best.copy()]
                for i in range(dim):
                    pt = x_best.copy()
                    step = 0.5 if x_best[i] < 0 else -0.5
                    pt[i] = np.clip(pt[i] + step, -5, 5)
                    simplex.append(pt)
                fvals_nm = np.array([func(np.clip(s, -5, 5)) for s in simplex])
                calls += dim
                alpha, gamma, rho = 1.0, 2.0, 0.5
                max_nm = 40
                for _ in range(max_nm):
                    if calls >= budget or time.time() > deadline - 5.0:
                        break
                    order = np.argsort(fvals_nm)
                    simplex = [simplex[i] for i in order]
                    fvals_nm = fvals_nm[order]
                    centroid = np.mean(simplex[:-1], axis=0)
                    # reflection
                    xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                    fr = func(xr)
                    calls += 1
                    if fr < fvals_nm[0]:
                        # expansion
                        xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                        fe = func(xe)
                        calls += 1
                        if fe < fr:
                            simplex[-1], fvals_nm[-1] = xe, fe
                        else:
                            simplex[-1], fvals_nm[-1] = xr, fr
                    elif fr < fvals_nm[-2]:
                        simplex[-1], fvals_nm[-1] = xr, fr
                    else:
                        # contraction
                        xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                        fc = func(xc)
                        calls += 1
                        if fc < fvals_nm[-1]:
                            simplex[-1], fvals_nm[-1] = xc, fc
                        else:
                            # shrink
                            for i in range(1, dim + 1):
                                simplex[i] = np.clip(simplex[0] + rho * (simplex[i] - simplex[0]), -5, 5)
                                fvals_nm[i] = func(simplex[i])
                                calls += 1
                    if fvals_nm[0] < f_best:
                        f_best = fvals_nm[0]
                        x_best = simplex[0].copy()
            except Exception:
                pass

        return x, fx
