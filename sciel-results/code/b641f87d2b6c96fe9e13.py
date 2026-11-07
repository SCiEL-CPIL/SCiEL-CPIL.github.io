import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def run_nelder_mead(x_start, max_calls):
            nonlocal calls, best_f, best_x
            local_best_x = x_start.copy()
            local_best_f = evaluate(x_start)
            local_calls = 1

            if local_calls >= max_calls or time.time() > deadline:
                return local_best_x, local_best_f, local_calls

            simplex = [x_start.copy()]
            simplex_f = [local_best_f]
            for i in range(dim):
                xi = x_start.copy()
                xi[i] += 0.5
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)
                f = evaluate(xi)
                local_calls += 1
                simplex_f.append(f)
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = xi.copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            min_step = 1e-12

            while local_calls < max_calls and time.time() < deadline:
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]

                if simplex_f[0] < local_best_f:
                    local_best_f = simplex_f[0]
                    local_best_x = simplex[0].copy()

                if local_calls >= max_calls - 1:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = evaluate(xr)
                local_calls += 1
                if fr < local_best_f:
                    local_best_f = fr
                    local_best_x = xr.copy()

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = evaluate(xe)
                    local_calls += 1
                    if fe < local_best_f:
                        local_best_f = fe
                        local_best_x = xe.copy()
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
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = evaluate(xc)
                    local_calls += 1
                    if fc < local_best_f:
                        local_best_f = fc
                        local_best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_f[i] = evaluate(simplex[i])
                            local_calls += 1
                            if simplex_f[i] < local_best_f:
                                local_best_f = simplex_f[i]
                                local_best_x = simplex[i].copy()
                        if np.max(np.abs(simplex[1] - simplex[0])) < min_step:
                            break

            return local_best_x, local_best_f, local_calls

        # ---------------- CMA-ES ----------------
        rng = np.random.default_rng(7)
        lam = 10
        mu = lam // 2

        # initial population: origin + diverse random points
        pop = [np.zeros(dim)]
        for _ in range(lam - 1):
            pop.append(rng.uniform(-4.5, 4.5, dim))
        pop = [np.clip(p, -5.0, 5.0) for p in pop]

        pop_f = [evaluate(p) for p in pop]
        total = calls

        sigma = 1.0
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        Cinv = np.eye(dim)
        pc = np.zeros(dim)
        cs = 1.0
        hs = 0.0
        c_c = 0.5
        c_sigma = 0.3
        c_cov = 0.5
        c_mu = 0.5

        # recompute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        def update_inv(A, delta, nu):
            # Sherman-Morrison-Woodbury for A + delta * v v^T
            t = 1.0 / (1.0 + delta * (v @ A @ v))
            return A - (delta * t) * np.outer(A @ v, A @ v)

        nu = 4 + 2.0 / (dim + 1.0)

        while total < budget - 60 and time.time() < deadline:
            # sort
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            # update mean
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * pop[i]
            x_mean = x_mean_new

            # update pc
            pc = (1 - cs) * pc + cs * ((np.sqrt(mu_eff) * sigma) / np.linalg.norm(
                np.array(pop[:mu]).T @ weights - x_mean) + 1e-300) * (
                np.array(pop[:mu]).T @ weights - x_mean)

            # hs
            hs = np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * (total + 1))) / (
                np.sqrt(dim) * (1 - c_sigma) + 1e-300)

            # update C
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = pop[i] - x_mean
                rank_mu += weights[i] * np.outer(diff, diff)
            rank_one = np.outer(pc, pc)
            C = (1 - c_cov) * C + c_cov * (rank_mu + (2 * hs - 1) * rank_one)
            C = 0.5 * (C + C.T)

            # update sigma
            sigma *= np.exp(c_sigma * (hs / c_sigma - 1) / mu_eff)
            sigma = min(max(sigma, 1e-12), 50.0)

            # recompute Cinv (5x5, cheap)
            try:
                Cinv = np.linalg.inv(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                Cinv = np.eye(dim)

            # generate new population
            new_pop = [x_mean]
            new_f = [evaluate(x_mean)]
            total = calls
            for _ in range(lam - 1):
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                z = rng.standard_normal(dim)
                y = L @ z
                x = x_mean + sigma * y
                x = np.clip(x, -5.0, 5.0)
                new_pop.append(x)
                new_f.append(evaluate(x))
                total = calls
                if total >= budget - 60 or time.time() > deadline:
                    break

            if total >= budget - 60 or time.time() > deadline:
                break

            pop = new_pop
            pop_f = new_f

        # ---------------- Nelder-Mead polish ----------------
        if calls < budget - 10 and time.time() < deadline:
            remaining = budget - calls
            run_nelder_mead(best_x, remaining)

        return best_x, best_f
