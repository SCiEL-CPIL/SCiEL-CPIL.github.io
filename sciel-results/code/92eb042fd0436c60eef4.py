"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset.

Strategy: hybrid global search + intensive (1+1)-ES polish with coordinate descent.
- Phase A: broad, cheap global sampling (Latin-hypercube + random) to locate the basin.
- Phase B: (1+1)-ES with 1/5th-rules sigma adaptation for smooth exploitation.
- Phase C: coordinate-descent + Nelder-Mead polish for the final push.
Early progress is weighted heavily by AOCC, so we front-load a solid global pass.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(2024)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def remaining_time():
            return deadline - time.time()

        def check():
            return calls_used < budget and remaining_time() > 15.0

        # ---------- Phase A: cheap global search ----------
        n_lh = 25
        n_rand = 15
        n_global = n_lh + n_rand

        lh = np.zeros((n_lh, dim))
        for d in range(dim):
            perm = rng.permutation(n_lh)
            lh[np.arange(n_lh), d] = (perm + rng.random(n_lh)) / n_lh
        lh = -5.0 + 10.0 * lh
        lh = np.clip(lh, -5, 5)

        rand_pts = rng.uniform(-4.5, 4.5, size=(n_rand, dim))
        candidates = np.vstack([lh, rand_pts])

        for i in range(candidates.shape[0]):
            if not check():
                break
            x = candidates[i]
            f = func(x)
            calls_used += 1
            update_best(x, f)

        # Local CMA-ES-lite around the best global point to refine the basin.
        if best_x is not None and check():
            calls_used = self._cmaes_lite(
                func, best_x.copy(), sigma_init=1.5,
                max_calls=min(80, budget - calls_used),
                deadline=deadline, rng=rng,
                calls_used=calls_used,
                update_best=update_best)

        # ---------- Phase B: (1+1)-ES polish ----------
        if best_x is not None and check():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            window = 15
            successes = 0
            window_count = 0
            max_polish = min(250, budget - calls_used)
            for _ in range(max_polish):
                if not check():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    successes += 1
                window_count += 1
                if window_count >= window:
                    p = successes / window_count
                    if p > 1.0 / (dim + 1) + 0.05:
                        sigma *= 1.1
                    elif p < 1.0 / (dim + 1) - 0.05:
                        sigma *= 0.9
                    successes = 0
                    window_count = 0
                sigma = np.clip(sigma, 1e-6, 5.0)

        # ---------- Phase C: coordinate descent ----------
        if best_x is not None and check():
            x = best_x.copy()
            fx = best_f
            step = 0.3
            cd_iters = min(40, budget - calls_used)
            for it in range(cd_iters):
                if not check():
                    break
                improved = False
                for d in range(dim):
                    if not check():
                        break
                    trial = x.copy()
                    trial[d] += step
                    trial = np.clip(trial, -5, 5)
                    ft = func(trial)
                    calls_used += 1
                    if ft < fx:
                        x, fx = trial, ft
                        improved = True
                        update_best(x, fx)
                        continue
                    trial2 = x.copy()
                    trial2[d] -= step
                    trial2 = np.clip(trial2, -5, 5)
                    ft2 = func(trial2)
                    calls_used += 1
                    if ft2 < fx:
                        x, fx = trial2, ft2
                        improved = True
                        update_best(x, fx)
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        # Final Nelder-Mead for the last push (cheap in 5D).
        if best_x is not None and check():
            x0 = best_x.copy()
            f0 = best_f
            delta = 0.2
            simplex = [x0.copy()]
            for d in range(dim):
                v = x0.copy()
                v[d] += delta
                v = np.clip(v, -5, 5)
                simplex.append(v)
            simplex_f = [f0]
            for v in simplex[1:]:
                if not check():
                    break
                fv = func(v)
                calls_used += 1
                simplex_f.append(fv)
                update_best(v, fv)
            max_nm = min(60, budget - calls_used)
            for _ in range(max_nm):
                if not check():
                    break
                if len(simplex) != dim + 1:
                    break
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                alpha = 1.0
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1
                update_best(xr, fr)
                if fr < simplex_f[0]:
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    update_best(xe, fe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    rho = 0.5
                    xc = centroid + rho * (worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    update_best(xc, fc)
                    if fc < worst_f:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls_used += 1
                            update_best(simplex[i], simplex_f[i])
                spread = np.linalg.norm(simplex[0] - simplex[-1])
                if spread < 1e-6:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f

    def _cmaes_lite(self, func, x_start, sigma_init, max_calls, deadline, rng,
                    calls_used, update_best):
        """A lightweight CMA-ES to refine the basin found by global search."""
        dim = self.dim
        budget = self.budget
        x_mean = x_start.copy()
        sigma = sigma_init
        lam = 8
        mu = lam // 2
        cmac = 1.0 / (dim + 4)
        cmam = (4 + dim / 4) / (dim + 4 + 3)
        c_sigma = (dim + 1.3) / (dim + 5.3 + 2 * lam / mu)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        local_calls = 0
        while local_calls < max_calls:
            if calls_used >= budget or (deadline - time.time()) < 20.0:
                break
            offspring = np.zeros((lam, dim))
            fitness = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget or (deadline - time.time()) < 20.0:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                offspring[i] = y
                f_val = func(y)
                calls_used += 1
                local_calls += 1
                fitness[i] = f_val
                update_best(y, f_val)
            if calls_used >= budget or (deadline - time.time()) < 20.0:
                break
            idx = np.argsort(fitness)
            offspring = offspring[idx]
            fitness = fitness[idx]
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * offspring[i]
            v = (x_new - x_mean) / sigma
            x_mean = x_new
            pc = (1 - cmac) * pc + cmac * v
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1 - (1 - cmac) ** 2) / chi_n <
                    (1.4 + 2 / (dim + 1)) * c_sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * v / chi_n
            delta_h = (1 - hsig) * c_sigma * (2 - c_sigma)
            C = ((1 - cmac * (2 - cmac) + cmac * cmam * delta_h) * C
                 + (cmac * (2 - cmac) - cmam * delta_h) * np.outer(pc, pc)
                 + c_sigma * (2 - c_sigma) * mueff * np.outer(ps, ps))
            C = (C + C.T) / 2
            eigvals, B = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)
            if sigma < 1e-8 or (fitness[0] - fitness[-1] < 1e-12 and local_calls > lam):
                break
        return calls_used
