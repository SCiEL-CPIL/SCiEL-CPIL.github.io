"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restart + Nelder-Mead + coordinate descent for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES
        def cma_es(x0, sigma0, max_calls, deadline):
            nonlocal calls, best_x, best_f
            lam = 8
            mu = lam // 2

            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            cs = (mu_eff + 2) / (lam + mu_eff + 5)
            cc = (4 + mu_eff / lam) / (lam + 4 + 2 * mu_eff / lam)
            c1 = 2 / (lam**2 + 18)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / (lam**2 + 18))
            ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (lam + 1)) - 1) + cs

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            # Evaluate initial mean
            fx = try_call(x_mean)
            if fx is None:
                return best_x, best_f, C, sigma

            stall_count = 0
            best_f_so_far = best_f

            while calls < max_calls and time.time() < deadline:
                # Generate offspring
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)

                normals = rng.standard_normal((lam, dim))
                scaled_normals = normals * np.sqrt(eigvals)
                zs = (eigvecs @ scaled_normals.T).T

                ys = []
                fs = []

                for i in range(lam):
                    y = x_mean + sigma * zs[i]
                    fy = try_call(y)
                    if fy is None:
                        break
                    ys.append(y)
                    fs.append(fy)

                if len(ys) < mu:
                    break

                idx_sorted = np.argsort(fs)
                ys = np.array(ys)
                fs = np.array(fs)
                ys = ys[idx_sorted]
                fs = fs[idx_sorted]

                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ys[0].copy()

                # Check for stall
                if fs[0] < best_f_so_far - 1e-10:
                    best_f_so_far = fs[0]
                    stall_count = 0
                else:
                    stall_count += 1

                # Restart if stalled
                if stall_count > 15 and sigma > 1e-4:
                    # Reset with smaller sigma
                    sigma *= 0.5
                    C = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    stall_count = 0
                    continue

                # Mean update
                y_mean = np.dot(weights, ys[:mu])
                z_mean = (y_mean - x_mean) / sigma

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_mean

                threshold = (1.4 + 2.0 / (dim + 1)) * np.sqrt(dim - 1)
                hsig = 1 if np.linalg.norm(ps) < threshold else 0

                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * z_mean

                zs_mu = (ys[:mu] - x_mean) / sigma
                H = np.zeros((dim, dim))
                for i in range(mu):
                    H += weights[i] * np.outer(zs_mu[i], zs_mu[i])

                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * H
                C = (C + C.T) / 2

                x_mean = y_mean

                if sigma < 1e-6:
                    break

            return best_x, best_f, C, sigma

        # Phase 2: Nelder-Mead
        def nelder_mead(x0, f0, max_calls, deadline):
            nonlocal calls, best_x, best_f

            simplex = [x0.copy()]
            simplex_f = [f0]

            # Initialize simplex with adaptive step based on local scale
            step = 0.3
            for i in range(dim):
                perturbation = x0.copy()
                perturbation[i] += step
                simplex.append(perturbation)

            for i in range(1, dim + 1):
                fx_i = try_call(simplex[i])
                if fx_i is None:
                    break
                simplex_f.append(fx_i)
                if fx_i < best_f:
                    best_f = fx_i
                    best_x = simplex[i].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            max_iterations = 150
            for _ in range(max_iterations):
                if calls >= max_calls or time.time() >= deadline:
                    break

                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                spread = max(simplex_f) - min(simplex_f)
                if spread < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_call(x_r)
                if f_r is None:
                    break

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = try_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = try_call(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = try_call(simplex[i])
                            if simplex_f[i] is None:
                                break
                        if simplex_f[-1] is None:
                            break

            return best_x, best_f

        # Phase 3: Coordinate descent
        def coordinate_descent(x0, max_calls, deadline):
            nonlocal calls, best_x, best_f
            x = x0.copy()
            f = best_f
            step = 0.1
            no_improve = 0

            for _ in range(200):
                if calls >= max_calls or time.time() >= deadline:
                    break
                if no_improve > 50:
                    break

                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_call(x_try)
                        if f_try is None:
                            return best_x, best_f
                        if f_try < f - 1e-12:
                            x = x_try
                            f = f_try
                            best_f = f
                            best_x = x.copy()
                            improved = True
                            no_improve = 0
                            break
                    if calls >= max_calls or time.time() >= deadline:
                        break

                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-8:
                        break
                else:
                    step = max(step * 1.2, 0.01)

            return best_x, best_f

        # Run phases
        # Phase 1: CMA-ES with 60% of budget
        cma_budget = int(0.6 * budget)
        best_x, best_f, C, sigma = cma_es(np.zeros(dim), 3.0, cma_budget, deadline)

        # Phase 2: Nelder-Mead with 30% of budget
        if best_x is not None and calls < budget and time.time() < deadline:
            nm_budget = int(0.3 * budget)
            best_x, best_f = nelder_mead(best_x, best_f, nm_budget, deadline)

        # Phase 3: Coordinate descent with remaining budget
        if best_x is not None and calls < budget and time.time() < deadline:
            cd_budget = budget
            best_x, best_f = coordinate_descent(best_x, cd_budget, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
