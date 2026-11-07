import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # CMA-ES parameters
        lam = 14
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def nelder_mead(x0, max_iter=80):
            nonlocal calls, best_x, best_f
            n = dim
            simplex = [x0.copy()]
            f0 = func(x0)
            calls += 1
            update_best(x0, f0)
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 4.5 else -0.5
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)
                fi = func(xi)
                calls += 1
                update_best(xi, fi)

            for it in range(max_iter):
                if not time_ok() or calls >= budget:
                    break
                # Sort simplex
                order = np.argsort([func_val for func_val in [f0] + [0]*n])  # placeholder
                # Actually we need to track fitness properly
                # Let's redo this more carefully

            # Simpler Nelder-Mead
            return

        def nelder_mead_v2(x0, max_iter=80):
            nonlocal calls, best_x, best_f
            n = dim
            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            calls += 1
            update_best(x0, fvals[0])

            for i in range(n):
                xi = x0.copy()
                step = 0.5 if abs(x0[i]) < 4.5 else 0.25
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)
                fi = func(xi)
                calls += 1
                fvals.append(fi)
                update_best(xi, fi)

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for it in range(max_iter):
                if not time_ok() or calls >= budget:
                    break

                # Sort
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func(xr)
                calls += 1
                update_best(xr, fr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func(xe)
                    calls += 1
                    update_best(xe, fe)
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
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func(xc)
                    calls += 1
                    update_best(xc, fc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi
                            update_best(simplex[i], fi)

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

        def cma_es(x0, sigma0, max_generations=None):
            nonlocal calls, best_x, best_f
            n = dim
            x = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            gen = 0

            if max_generations is None:
                max_generations = (budget - calls) // lam

            for gen in range(max_generations):
                if not time_ok() or calls >= budget:
                    break

                # Generate population
                candidates = []
                for i in range(lam):
                    if not time_ok() or calls >= budget:
                        break
                    z = np.random.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    candidates.append((fy, y))
                    update_best(y, fy)

                if len(candidates) < mu:
                    break

                # Sort
                candidates.sort(key=lambda t: t[0])
                y_w = np.array([c[1] for c in candidates[:mu]])

                # Weighted mean
                m = np.sum(weights[:, None] * y_w, axis=0)

                # Update evolution paths
                old_sigma = sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m - x)))

                # Delta_h check
                norm_ps = np.linalg.norm(ps)
                # Approximate the threshold
                threshold = (1.4 + 2 / (n + 1)) * chi_n
                # More accurate: use the path length normalization
                # For simplicity, use the standard check
                if norm_ps < threshold:
                    delta_h = 0
                else:
                    delta_h = 1

                pc = (1 - c_c) * pc + c_c * (m - x) / sigma

                # Covariance matrix update
                C = B @ np.diag(D**2) @ B.T
                C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * np.outer(pc, pc)

                # Eigen decomposition
                D_sq, B = np.linalg.eigh(C)
                D_sq = np.clip(D_sq, 1e-12, None)
                D = np.sqrt(D_sq)

                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

                # Update mean
                x = m

        # Strategy: Use CMA-ES with restarts, interleaved with Nelder-Mead polishing
        # Phase 1: Global exploration with CMA-ES
        # Phase 2: Local refinement with Nelder-Mead

        # Start with a few random points to get a good initial guess
        n_random = min(10, budget)
        for i in range(n_random):
            if not time_ok() or calls >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            calls += 1
            update_best(x, f)

        # Run CMA-ES from the best point
        cma_es(best_x, 2.0)

        # If we still have budget, do Nelder-Mead polishing
        if time_ok() and calls < budget:
            nelder_mead_v2(best_x, max_iter=80)

        # Final check: if we have more budget, try another CMA-ES restart from best
        if time_ok() and calls < budget and (budget - calls) >= lam * 2:
            cma_es(best_x, 1.0)

        if time_ok() and calls < budget:
            nelder_mead_v2(best_x, max_iter=50)

        return best_x, best_f
