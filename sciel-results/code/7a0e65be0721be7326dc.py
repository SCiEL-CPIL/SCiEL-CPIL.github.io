import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_cmaes(x_start, sigma_start, budget_cma, rng_cma):
            nonlocal best_x, best_f, calls
            if calls >= budget_cma or time.time() >= deadline:
                return

            lam = max(4, int(4 + 3 * dim))
            mu = lam // 2
            w = np.log(mu + 1.0) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            cs = (mu_eff + 2) / (lam + mu_eff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
            cc = (4 + mu_eff / lam) / (lam + 4 + 2 * mu_eff / lam)
            c1 = 2 / ((dim + 1.3) ** 2 - mu_eff)
            cmu = min(1 - c1,
                      2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 - mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            x = np.asarray(x_start, dtype=np.float64).copy()
            sigma = float(sigma_start)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            x_best = x.copy()
            f_best = np.inf

            # evaluate starting point
            fx = try_call(x)
            if fx is None:
                return
            f_best = fx
            x_best = x.copy()

            stall = 0
            stall_tol = 1e-10
            stall_cap = 5 * lam

            while calls < budget_cma and time.time() < deadline:
                # generate population
                z = rng_cma.standard_normal((lam, dim))
                pop = np.einsum('ij,jk,ik->ik', B, D, z) * sigma + x
                fvals = np.empty(lam)
                for k in range(lam):
                    fk = try_call(pop[k])
                    if fk is None:
                        return
                    fvals[k] = fk

                order = np.argsort(fvals)
                # update best
                if fvals[order[0]] < f_best:
                    f_best = fvals[order[0]]
                    x_best = pop[order[0]].copy()

                # mean update (recombination)
                y = np.einsum('ij,jk,ik->ik', B, D, z[order[:mu]])
                mean_shift = sigma * np.sum(w[:, None] * y, axis=0)
                x = x + mean_shift

                # covariance adaptation
                yw = np.einsum('ij,jk,ik->ik', B, D, z[order[:mu]])
                yw = yw - (1 - c1) * pc
                delta_s = 1.0 if fvals[order[0]] < f_best - 1e-16 else 0.0
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * yw / chi_n
                ps_norm = np.linalg.norm(ps)
                C = ((1 - c1 - cmu) * C
                     + c1 * (np.outer(pc, pc) + (1 - (1 - cs) ** 2) * np.eye(dim))
                     + cmu * (yw.T @ yw / mu_eff))
                pc = (1 - cc) * pc + cc * mean_shift / sigma

                # sigma adaptation
                sigma *= np.exp((cs / ds) * (ps_norm / chi_n - 1.0))
                if sigma > 1e6:
                    sigma = 1e6

                # periodic re-orthogonalization of C
                if (calls % (10 * lam)) < lam:
                    ev, evec = np.linalg.eigh(C)
                    ev = np.maximum(ev, 1e-20)
                    C = evec @ (ev[:, None] * evec.T)
                    B, D = evec, np.sqrt(ev)

                # stall detection -> adaptive restart
                stall_tol_eff = max(stall_tol, 1e-8 * abs(f_best))
                if fvals[order[0]] > f_best - stall_tol_eff:
                    stall += 1
                else:
                    stall = 0
                if stall >= stall_cap or sigma < 1e-12:
                    break

            return x_best

        def coord_descent(x_start, budget_cd, rng_cd):
            nonlocal best_x, best_f, calls
            x = np.asarray(x_start, dtype=np.float64).copy()
            f = try_call(x)
            if f is None:
                return
            step = 1.0
            no_improve = 0
            max_no_improve = 200
            while calls < budget_cd and time.time() < deadline and no_improve < max_no_improve:
                improved = False
                for i in range(dim):
                    for s in (step, -step):
                        xn = x.copy()
                        xn[i] += s
                        fn = try_call(xn)
                        if fn is None:
                            return
                        if fn < f:
                            x = xn
                            f = fn
                            improved = True
                            break
                if improved:
                    no_improve = 0
                    step *= 0.7
                else:
                    no_improve += 1
                    step *= 0.5
                if step < 1e-9:
                    break
            return x

        # ---- Phase 1: random initialization ----
        x0 = np.zeros(dim)
        try_call(x0)
        for _ in range(min(10, budget - 1)):
            if calls >= budget:
                break
            try_call(rng.uniform(-4, 4, dim))

        # ---- Phase 2: CMA-ES with adaptive restarts ----
        n_restarts = 3
        budget_cma = int(0.85 * budget)
        for r in range(n_restarts):
            if calls >= budget_cma or time.time() >= deadline:
                break
            if r == 0:
                x_start = best_x.copy()
                sigma_start = 2.0
            else:
                # restart around the current best with a fresh, moderate sigma
                x_start = best_x + rng.normal(0, 0.5, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                sigma_start = 1.0
            run_cmaes(x_start, sigma_start, budget_cma, rng)

        # ---- Phase 3: coordinate-descent polish ----
        if calls < budget and time.time() < deadline:
            coord_descent(best_x.copy(), budget - calls, rng)

        return best_x, best_f
