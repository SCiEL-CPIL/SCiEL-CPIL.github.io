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

        def run_11es(x_start, sigma_start, budget_11, rng_11):
            nonlocal best_x, best_f, calls

            x = x_start.copy()
            sigma = sigma_start
            c_sigma = 1.0 / (dim + 1)
            c_mu = 1.0 / (dim + 1)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            fx = try_call(x)
            if fx is None:
                return

            while calls < budget_11 and time.time() < deadline:
                z = rng_11.standard_normal(dim)
                y = x + sigma * z
                fy = try_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= np.exp(c_sigma * (1.0 / chi_n - 1.0))
                else:
                    sigma *= np.exp(-c_sigma * (1.0 / chi_n))

                if sigma < 1e-8:
                    break

            return x, sigma

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

        n_random = min(15, budget - 1)
        for _ in range(n_random):
            if calls >= budget:
                break
            xr = rng.uniform(-4, 4, dim)
            fr = try_call(xr)

        # Phase 2: (1+1)-ES for global optimization
        budget_11 = int(0.65 * budget)
        x_start = best_x.copy()
        sigma_start = 2.0

        run_11es(x_start, sigma_start, budget_11, rng)

        # Phase 3: Nelder-Mead for local refinement
        if calls < budget and time.time() < deadline:
            budget_nm = budget - calls
            x_start = best_x.copy()
            run_nelder_mead(x_start, budget_nm, rng)

        return best_x, best_f
