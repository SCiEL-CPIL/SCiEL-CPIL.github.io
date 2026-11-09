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
        rng = np.random.default_rng(42)

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

            lam = 10
            mu = lam // 2

            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            cs = (mu_eff + 2) / (lam + mu_eff + 5)
            cc = (4 + mu_eff / lam) / (lam + 4 + 2 * mu_eff / lam)
            c1 = 2 / (lam ** 2 + 18)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / (lam ** 2 + 18))
            ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (lam + 1)) - 1) + cs

            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            fx = try_call(x_mean)
            if fx is None:
                return

            while calls < budget_cma and time.time() < deadline:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)

                normals = rng_cma.standard_normal((lam, dim))
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
                ys_arr = np.array(ys)
                fs_arr = np.array(fs)
                ys_sel = ys_arr[idx_sorted]
                fs_sel = fs_arr[idx_sorted]

                if fs_sel[0] < best_f:
                    best_f = fs_sel[0]
                    best_x = ys_sel[0].copy()

                y_mean = np.dot(weights, ys_sel[:mu])
                z_mean = (y_mean - x_mean) / sigma

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_mean

                threshold = (1.4 + 2.0 / (dim + 1)) * np.sqrt(dim - 1)
                hsig = 1 if np.linalg.norm(ps) < threshold else 0

                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * z_mean

                zs_mu = (ys_sel[:mu] - x_mean) / sigma
                H = np.zeros((dim, dim))
                for i in range(mu):
                    H += weights[i] * np.outer(zs_mu[i], zs_mu[i])

                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * H
                C = (C + C.T) / 2

                x_mean = y_mean

                if sigma < 1e-7:
                    break

                if np.max(np.abs(x_mean)) > 4.5 and sigma < 0.01:
                    break

            return x_mean, sigma

        def run_nelder_mead(x_start, budget_nm, rng_nm):
            nonlocal best_x, best_f, calls

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            simplex = [x_start.copy()]
            f_vals = [best_f]

            for i in range(dim):
                x_i = x_start.copy()
                x_i[i] += sigma_nm
                f_i = try_call(x_i)
                if f_i is None:
                    return
                simplex.append(x_i)
                f_vals.append(f_i)

            simplex = np.array(simplex)
            f_vals = np.array(f_vals)

            max_iter = budget_nm // (dim + 1)
            for it in range(max_iter):
                if calls >= budget_nm or time.time() >= deadline:
                    break

                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]

                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_call(x_r)
                if f_r is None:
                    break

                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = try_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = try_call(x_c)
                    if f_c is None:
                        break
                    if f_c < f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_shrink = try_call(simplex[i])
                            if f_shrink is None:
                                return
                            f_vals[i] = f_shrink

                if np.max(np.abs(simplex - simplex[0])) < 1e-8:
                    break

            return

        # Phase 1: Brief random search to find a good starting point
        x0 = np.zeros(dim)
        f0 = try_call(x0)

        n_random = min(20, budget - 1)
        for _ in range(n_random):
            if calls >= budget:
                break
            xr = rng.uniform(-4, 4, dim)
            fr = try_call(xr)
            if fr < best_f:
                x0 = best_x.copy()

        # Phase 2: CMA-ES for global optimization
        budget_cma = int(0.75 * budget)
        x_start = best_x.copy()
        sigma_start = 2.0

        run_cmaes(x_start, sigma_start, budget_cma, rng)

        # Phase 3: Nelder-Mead for local refinement
        if calls < budget and time.time() < deadline:
            budget_nm = budget - calls
            x_start = best_x.copy()
            run_nelder_mead(x_start, budget_nm, rng)

        return best_x, best_f
