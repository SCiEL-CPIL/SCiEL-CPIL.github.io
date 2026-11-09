"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton-seed Nelder-Mead with adaptive ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(index, dim):
            bases = [2, 3, 5, 7, 11]
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = -5.0 + 10.0 * halton(index + 1, bases[d])
            return x

        def nelder_mead(x0, max_iters, init_step=0.5):
            nonlocal best_x, best_f, calls_made
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                step = init_step
                if abs(x0[i]) > 4.5:
                    step = min(step, 0.3)
                    simplex[i + 1][i] += -step if x0[i] > 0 else step
                else:
                    simplex[i + 1][i] += step if x0[i] < 0 else -step

            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fx = eval_func(simplex[i])
                if fx is None:
                    return
                fvals[i] = fx

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            min_step = 1e-12

            for _ in range(max_iters):
                if time.time() > deadline:
                    return
                if calls_made >= budget - 2:
                    return

                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = eval_func(xr)
                if fr is None:
                    return

                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = eval_func(xe)
                    if fe is None:
                        return
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = centroid + 0.5 * (xr - centroid)
                        fc = eval_func(xc)
                        if fc is None:
                            return
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fvals[i] = eval_func(simplex[i])
                                if fvals[i] is None:
                                    return
                    else:
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        fc = eval_func(xc)
                        if fc is None:
                            return
                        if fc <= fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                fvals[i] = eval_func(simplex[i])
                                if fvals[i] is None:
                                    return

                if np.max(fvals) - np.min(fvals) < min_step:
                    return

        def local_es(x_start, fx_start, max_steps, sigma_init=0.1):
            nonlocal best_x, best_f, calls_made
            x = x_start.copy()
            fx = fx_start
            sigma = sigma_init
            rng_local = np.random.default_rng(12345)
            for _ in range(max_steps):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 1:
                    break
                y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                fy = eval_func(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 1: Halton sequence for deterministic space-filling coverage
        # 40 evaluations to systematically explore the 5D domain
        n_halton = min(40, max(10, budget // 12))
        for i in range(n_halton):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break
            x = halton_point(i, dim)
            x = np.clip(x, -5, 5)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return None, float('inf')

        # Phase 2: Nelder-Mead from the best Halton point
        if calls_made < budget - 10:
            nelder_mead(best_x.copy(), max_iters=100, init_step=0.5)

        # Phase 2b: Additional NM restarts from other good Halton points
        # Pick the next few best points from Halton sequence for diversity
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break
            # Use Halton points at different indices for diversity
            x_start = halton_point(n_halton + r * 5, dim)
            x_start = np.clip(x_start, -5, 5)
            nelder_mead(x_start, max_iters=60, init_step=0.4)

        # Phase 3: Local ES refinement
        if best_x is not None and calls_made < budget - 5:
            local_es(best_x.copy(), best_f, max_steps=15, sigma_init=0.2)

        if best_x is not None and calls_made < budget - 3:
            local_es(best_x.copy(), best_f, max_steps=20, sigma_init=0.05)

        # Phase 4: Final NM polish
        if best_x is not None and calls_made < budget - 5:
            nelder_mead(best_x.copy(), max_iters=40, init_step=0.1)

        return best_x, best_f
# EVOLVE-BLOCK-END
