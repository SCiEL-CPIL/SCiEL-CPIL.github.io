"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and structured initial sampling for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_fx, best_x
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        def cma_es(x0, sigma0, max_evals, rng_seed=None):
            nonlocal calls_made, best_x, best_fx

            if rng_seed is not None:
                local_rng = np.random.default_rng(rng_seed)
            else:
                local_rng = rng

            # CMA-ES parameters tuned for dim=5
            lam = 9
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            c_sigma = (mueff + 2) / (d + mueff + 5)
            c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            cs = (c_sigma + 1) * (0.5 + 0.5 * min(1, mueff / d - 1))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            x = np.array(x0, dtype=float).copy()
            sigma = sigma0
            ps = np.zeros(d)
            pc = np.zeros(d)
            C = np.eye(d)

            # Precompute inverse square root
            def invsqrtC_update(C_mat):
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-20, None)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            invsqrtC = invsqrtC_update(C)

            # Initial population
            pop = np.tile(x, (lam, 1)) + sigma * local_rng.standard_normal((lam, d))
            pop = np.clip(pop, -5, 5)
            fvals = np.array([call_func(xi) for xi in pop])

            while calls_made < max_evals and time.time() < deadline:
                if calls_made + lam > max_evals:
                    break

                # Sort and select
                order = np.argsort(fvals)
                selected = pop[order[:mu]]

                # Update mean
                x_new = np.sum(selected * w[:, None], axis=0)

                # Update sigma path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (invsqrtC @ (x_new - x))
                norm_ps = np.linalg.norm(ps)
                hsig = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update C path
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new - x)
                norm_pc = np.linalg.norm(pc)
                hsig_c = 1 if norm_pc / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update covariance
                delta = hsig_c * c_1 * (1 - c_1) * np.outer(pc, pc)
                rank_mu = np.sum([w[i] * np.outer(selected[i] - x, selected[i] - x) for i in range(mu)], axis=0)
                C = (1 - c_1 - c_mu) * C + delta + c_mu * rank_mu

                # Symmetrize for numerical stability
                C = 0.5 * (C + C.T)

                x = x_new

                # Update sigma
                sigma *= np.exp(c_sigma / damps * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50.0)

                # Update inverse sqrt periodically (every generation for dim=5)
                invsqrtC = invsqrtC_update(C)

                # Generate new population
                pop = np.tile(x, (lam, 1)) + sigma * local_rng.standard_normal((lam, d))
                pop = np.clip(pop, -5, 5)
                fvals = np.array([call_func(xi) for xi in pop])

            return best_x, best_fx

        def local_search(x0, max_evals, step=0.3):
            """Simple coordinate descent / Nelder-Mead style local refinement."""
            nonlocal calls_made, best_x, best_fx

            x = np.array(x0, dtype=float).copy()
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Nelder-Mead simplex
            simplex = [x.copy()]
            for i in range(d):
                xi = x.copy()
                xi[i] += step
                simplex.append(xi)

            fvals = np.array([call_func(s) for s in simplex])

            for _ in range(max_evals // (d + 2)):
                if calls_made >= max_evals or time.time() > deadline:
                    break

                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = fvals[idx]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = call_func(xr)

                if fxr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = call_func(xe)
                    if fxe < fxr:
                        simplex[-1] = xe
                        fvals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                elif fxr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fxr
                else:
                    # Contraction
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = call_func(xc)
                    if fxc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = call_func(simplex[i])

            return best_x, best_fx

        # Phase 1: Structured initial sampling
        n_initial = 30
        candidates = []

        # Grid-like points
        grid_vals = [-4, -2, 0, 2, 4]
        for i in range(min(n_initial // 3, budget)):
            if time.time() > deadline or calls_made >= budget:
                break
            x = np.array([grid_vals[(i // (d // 2 + 1)) % len(grid_vals)] + 0.5 * rng.standard_normal() for _ in range(d)])
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            candidates.append((fx, x.copy()))

        # Random points
        for i in range(n_initial - len(candidates)):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            fx = call_func(x)
            candidates.append((fx, x.copy()))

        # Origin and axis points
        origin = np.zeros(d)
        fx_o = call_func(origin)
        candidates.append((fx_o, origin.copy()))

        for axis in range(min(d, budget - calls_made)):
            if time.time() > deadline or calls_made >= budget:
                break
            for sign in [1, -1]:
                if calls_made >= budget:
                    break
                x = sign * 2.0 * np.zeros(d)
                x[axis] = sign * 2.0
                fx = call_func(x)
                candidates.append((fx, x.copy()))

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)
            candidates.append((best_fx, best_x.copy()))

        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from best candidates with adaptive budget
        remaining = budget - calls_made
        n_restarts = min(5, max(1, remaining // 60))
        base_budget = remaining // n_restarts

        for r in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break

            if r == 0:
                x0 = candidates[0][1].copy()
                sigma0 = 1.5
            elif r < len(candidates):
                x0 = candidates[r][1].copy()
                sigma0 = 1.0
            else:
                x0 = best_x + 0.8 * rng.standard_normal(d)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.8

            max_evals = calls_made + base_budget
            cma_es(x0, sigma0, max_evals, rng_seed=1000 + r)

        # Phase 3: Local refinement around best point
        if calls_made < budget and time.time() < deadline:
            remaining_ls = min(budget - calls_made, 50)
            local_search(best_x, calls_made + remaining_ls, step=0.5)

        # Phase 4: Final CMA-ES polish if budget remains
        if calls_made < budget - 10 and time.time() < deadline:
            remaining_polish = budget - calls_made
            if remaining_polish >= 20:
                cma_es(best_x, 0.3, budget, rng_seed=9999)

        return best_x, best_fx
# EVOLVE-BLOCK-END
