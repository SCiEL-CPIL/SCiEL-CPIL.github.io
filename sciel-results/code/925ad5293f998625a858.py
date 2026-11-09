"""Optimizer for BBOB f13 (Sharp ridge) on [-5,5]^5."""
import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before the 330s wall-clock kill

        rng = np.random.default_rng(12345)

        # ---- CMA-ES ----
        x = np.zeros(dim)
        sigma = 2.0
        popsize = max(4 + int(3 * np.log(dim)), dim + 2)

        C = np.eye(dim)
        pc = np.zeros(dim)
        cs = 0.0

        mu = popsize // 2
        w = np.log(popsize + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        cc = (4 + mueff / dim) / ((dim + 4) ** 2 + mueff / 2)
        cs_ = (1 + 2 * max(0.0, np.sqrt((mueff - 1) / (dim + 1)) - 1)) / (
            dim + 1 + 2 * max(0.0, mueff / 2 - 1))
        cmu = min(2 - cc, 0.5 * (mueff - 2) / (dim + mueff - 2))

        calls = 0
        fx = func(x)
        calls += 1
        best_x, best_f = x.copy(), fx

        def _sample():
            w_eig, v_eig = np.linalg.eigh(C)
            w_eig = np.maximum(w_eig, 1e-12)
            Z = rng.standard_normal((popsize, dim))
            return x + (Z * np.sqrt(w_eig)[None, :]) @ v_eig.T * sigma

        ys = _sample()
        fs = np.array([func(y) for y in ys])
        calls += popsize

        for it in range(5000):
            if calls >= budget or time.time() > deadline:
                break

            order = np.argsort(fs)
            ysel = ys[order[:mu]]
            fsel = fs[order[:mu]]
            xmean = np.sum(w[:, None] * ysel, axis=0)

            gbest = int(np.argmin(fs))
            if fs[gbest] < best_f:
                best_f = fs[gbest]
                best_x = ys[gbest].copy()

            ydiff = (xmean - x) / sigma
            pc = (1 - cc) * pc + cc * ydiff
            cs = (1 - cs_) * cs + np.sqrt(mueff * cc * (2 - cc)) * np.linalg.norm(pc)

            C = (1 - (mu * cmu + 2 * cmu * cc * (1 - cc))) * C \
                + (cmu / (sigma * sigma)) * np.sum(w[:, None] * np.outer(ysel - x, ysel - x), axis=0) \
                + (cc * (2 - cc)) * np.outer(pc, pc)
            C = 0.5 * (C + C.T)
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-8 * np.eye(dim)

            norm_pc = np.linalg.norm(pc)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(cs_ / (dim + 2) * (norm_pc / chi_n - 1))
            sigma = min(max(sigma, 1e-10), 10.0)

            x = xmean

            ys = _sample()
            remaining = budget - calls
            if remaining <= 0:
                break
            n = min(popsize, remaining)
            ys = ys[:n]
            fs = np.array([func(y) for y in ys])
            calls += n

        # ---- Nelder-Mead polish ----
        try:
            x0 = best_x.copy()
            f0 = best_f
            simplex = [x0.copy()]
            for i in range(dim):
                p = x0.copy()
                p[i] += 0.05
                simplex.append(p)
            simplex = np.array(simplex)
            fvals = np.array([f0] + [func(p) for p in simplex[1:]])
            calls += dim

            for _ in range(3000):
                if calls >= budget or time.time() > deadline:
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                centroid = np.mean(simplex[1:-1], axis=0)
                xr = centroid + 1.0 * (centroid - simplex[-1])
                fr = func(np.clip(xr, -5, 5))
                calls += 1
                if fr < fvals[-2]:
                    if fr < fvals[0]:
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        fe = func(np.clip(xe, -5, 5))
                        calls += 1
                        if fe < fr:
                            simplex[-1] = np.clip(xe, -5, 5)
                            fvals[-1] = fe
                        else:
                            simplex[-1] = np.clip(xr, -5, 5)
                            fvals[-1] = fr
                    else:
                        simplex[-1] = np.clip(xr, -5, 5)
                        fvals[-1] = fr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    fc = func(np.clip(xc, -5, 5))
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = np.clip(xc, -5, 5)
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(np.clip(simplex[i], -5, 5))
                            calls += 1
                            if calls >= budget:
                                break
                        if calls >= budget:
                            break
        except Exception:
            pass

        return best_x, best_f
