"""Optimizer for BBOB f17 (Schaffer's F7) using island-based CMA-ES with Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-based CMA-ES with Nelder-Mead local refinement for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def nelder_mead(x0, max_calls, step0=0.5):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_x, best_fx, calls_made
            n = d
            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += step0
                simplex.append(np.clip(xi, -5, 5))
            fvals = [call_func(s) for s in simplex]
            idx = np.argmin(fvals)
            if fvals[idx] < best_fx:
                best_fx = fvals[idx]
                best_x = simplex[idx].copy()

            alpha, gamma, rho = 1.0, 2.0, 0.5
            for it in range(max_calls // (n + 1) + 1):
                if not time_ok() or calls_made >= max_calls:
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = call_func(x_r)
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = call_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = call_func(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = call_func(simplex[i])
                idx = np.argmin(fvals)
                if fvals[idx] < best_fx:
                    best_fx = fvals[idx]
                    best_x = simplex[idx].copy()
                if np.max(np.abs(fvals - fvals[0])) < 1e-10:
                    break
            return best_x, best_fx

        def cma_es_island(x0, sigma0, max_calls, rng_local, island_id):
            """Single CMA-ES island."""
            nonlocal best_x, best_fx, calls_made
            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1 / np.sum(w ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (1 - cc - 2 * (min(1, cs - 1) * (n - 1) / n))
            cmu = (2 / ((n + 2) ** 2 + mueff)) * (mueff - 2 + 2 / (n - 2)) - c1
            cmu = min(cmu, 1 - c1)

            sigma = sigma0
            mean = x0.copy()
            C = np.eye(n)
            C_eig = np.linalg.eigh(C)
            C_eigvals = C_eig[0]
            C_eigvecs = C_eig[1]
            C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals)) @ C_eigvecs.T

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial population
            pop = []
            fvals = []
            for i in range(lam):
                z = rng_local.standard_normal(n)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                fx = call_func(x)
                pop.append(x)
                fvals.append(fx)

            idx = np.argmin(fvals)
            if fvals[idx] < best_fx:
                best_fx = fvals[idx]
                best_x = pop[idx].copy()

            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            max_iter = 80
            for iteration in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                old_mean = mean.copy()
                mean = sum(w[i] * pop[i] for i in range(mu))

                if np.linalg.norm(C_sqrt.T @ (mean - old_mean) / sigma) > (1.4 + 1 / cc) * cs * chi_n:
                    pc = (1 - cs) * pc
                else:
                    pc = (1 - cs) * pc + cs / chi_n * (C_sqrt.T @ (mean - old_mean)) / sigma

                hsig = (np.sqrt((1 - (1 - cc) ** (2 * (iteration + 1))) / (1 - (1 - cc) ** 2)) - 1 + cc) * chi_n
                if np.linalg.norm(pc) / chi_n < (1.4 + 1 / cc) * hsig:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (mean - old_mean) / sigma
                else:
                    ps = (1 - cs) * ps

                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y = (pop[i] - old_mean) / sigma
                    rank_mu += w[i] * np.outer(y, y)

                C = (1 - cs * c1 - cmu) * C + (1 - cs) * c1 * np.outer(pc, pc) + cmu * rank_mu
                C = (C + C.T) / 2
                try:
                    C_eig = np.linalg.eigh(C)
                    C_eigvals = np.maximum(C_eig[0], 1e-12)
                    C_eigvecs = C_eig[1]
                    C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
                    C_sqrt = C_eigvecs @ np.diag(np.sqrt(C_eigvals))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)

                sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                if np.linalg.norm(mean - old_mean) < 1e-10 * max(1, sigma):
                    break

                new_pop = []
                new_fvals = []
                for i in range(lam):
                    z = rng_local.standard_normal(n)
                    x = mean + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    fx = call_func(x)
                    new_pop.append(x)
                    new_fvals.append(fx)

                idx = np.argmin(new_fvals)
                if new_fvals[idx] < best_fx:
                    best_fx = new_fvals[idx]
                    best_x = new_pop[idx].copy()

                order = np.argsort(new_fvals)
                pop = [new_pop[i] for i in order]
                fvals = [new_fvals[i] for i in order]

            return best_x, best_fx

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for i in range(d):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        for _ in range(8):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: Island-based CMA-ES
        n_islands = 3
        base_calls = max(20, (budget - calls_made - 30) // n_islands)

        for island_id in range(n_islands):
            if not time_ok():
                break
            if island_id == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                x0 = best_x + rng.normal(0, 1.5, d)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 1.5

            island_rng = np.random.default_rng(42 + island_id)
            max_calls = calls_made + base_calls
            cma_es_island(x0, sigma0, max_calls, island_rng, island_id)

        # Phase 3: Nelder-Mead refinement on best solution
        if time_ok() and calls_made < budget:
            nm_calls = min(50, budget - calls_made)
            nelder_mead(best_x.copy(), nm_calls, step0=0.5)

        # Phase 4: Final CMA-ES polish with small sigma
        if time_ok() and calls_made < budget:
            remaining = budget - calls_made
            if remaining > 15:
                polish_rng = np.random.default_rng(99)
                max_calls = calls_made + min(remaining, 40)
                cma_es_island(best_x.copy(), 0.3, max_calls, polish_rng, 99)

        # Phase 5: Final Nelder-Mead polish
        if time_ok() and calls_made < budget:
            nm_calls = min(20, budget - calls_made)
            nelder_mead(best_x.copy(), nm_calls, step0=0.1)

        return best_x, best_fx
# EVOLVE-BLOCK-END
