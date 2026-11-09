"""Optimizer for BBOB f17 (Schaffer's F7) using CMA-ES with Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Nelder-Mead refinement and multi-restart for Schaffer's F7."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget:
                return best_fx
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def nelder_mead(x0, max_evals, alpha=1.0, gamma=2.0, rho=0.5, sigma=1.0):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal calls_made, best_x, best_fx

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(d):
                xi = x0.copy()
                xi[i] += sigma
                simplex.append(np.clip(xi, -5, 5))

            fvals = np.array([call_func(s) for s in simplex])
            simplex = np.array(simplex)

            local_best_idx = np.argmin(fvals)
            local_best_x = simplex[local_best_idx].copy()
            local_best_f = fvals[local_best_idx]

            for _ in range(max_evals):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                local_best_idx = 0
                local_best_x = simplex[0].copy()
                local_best_f = fvals[0]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
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
                    xc = np.clip(xc, -5, 5)
                    fc = call_func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + rho * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = call_func(simplex[i])
                        if calls_made >= budget or time.time() > deadline:
                            break

                if fvals[0] < local_best_f:
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()

            return local_best_x, local_best_f

        def cma_es(x0, sigma0, max_evals):
            """CMA-ES with adaptive population size."""
            nonlocal calls_made, best_x, best_fx

            # CMA-ES parameters for dim=5
            lam = 9  # population size
            mu = lam // 2  # number of selected individuals
            weights = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1)).sum()
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            c_sigma = (mueff + 2) / (d + mueff + 5)
            c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            cs = (c_sigma + 1) * (0.5 + 0.5 * min(1, mueff / d - 1))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            ps = np.zeros(d)
            pc = np.zeros(d)
            C = np.eye(d)
            invsqrtC = np.eye(d)

            local_best_x = x.copy()
            local_best_f = np.inf

            # Generate initial population
            pop = np.tile(x, (lam, 1)) + sigma * rng.standard_normal((lam, d))
            pop = np.clip(pop, -5, 5)
            fvals = np.array([call_func(xi) for xi in pop])

            idx_best = np.argmin(fvals)
            local_best_f = fvals[idx_best]
            local_best_x = pop[idx_best].copy()
            if local_best_f < best_fx:
                best_fx = local_best_f
                best_x = local_best_x.copy()

            while calls_made < max_evals and time.time() < deadline:
                # Check if we have enough budget for next generation
                if calls_made + lam > max_evals:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                selected = pop[order[:mu]]

                # Update mean
                x_new = weights @ selected

                # Update sigma control vector
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (
                    invsqrtC @ (x_new - x)
                )

                # Check hsig
                norm_ps = np.linalg.norm(ps)
                hsig = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update path for C
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new - x)

                # Check hsig for C update
                norm_pc = np.linalg.norm(pc)
                hsig_c = 1 if norm_pc / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update covariance matrix
                delta = hsig_c * c_1 * (1 - c_1) * np.outer(pc, pc)
                diff = selected - x
                rank_mu_term = np.sum([weights[i] * np.outer(diff[i], diff[i]) for i in range(mu)], axis=0)
                C = (1 - c_1 - c_mu) * C + delta + c_mu * rank_mu_term

                x = x_new

                # Update sigma
                sigma *= np.exp(c_sigma / damps * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update inverse square root of C periodically
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                invsqrtC = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                # Generate new population
                pop = np.tile(x, (lam, 1)) + sigma * rng.standard_normal((lam, d))
                pop = np.clip(pop, -5, 5)
                fvals = np.array([call_func(xi) for xi in pop])

                # Track best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < local_best_f:
                    local_best_f = fvals[idx_best]
                    local_best_x = pop[idx_best].copy()
                    if local_best_f < best_fx:
                        best_fx = local_best_f
                        best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Structured initial sampling
        n_initial = min(20, budget)
        candidates = []

        # Sample from multiple scales
        for scale in [1.0, 2.0, 3.0, 4.0]:
            n_at_scale = max(3, n_initial // 4)
            for i in range(n_at_scale):
                if time.time() > deadline or calls_made >= budget:
                    break
                x = rng.uniform(-5, 5, d) * scale
                x = np.clip(x, -5, 5)
                fx = call_func(x)
                candidates.append((fx, x.copy()))
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)
            candidates.append((best_fx, best_x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES from top candidates with Nelder-Mead refinement
        remaining_budget = budget - calls_made
        n_restarts = min(3, max(1, remaining_budget // 100))
        budget_per_restart = remaining_budget // n_restarts

        for restart in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break

            # Pick a starting point
            if restart == 0:
                x0 = candidates[0][1].copy()
            elif restart < len(candidates):
                x0 = candidates[min(restart, len(candidates) - 1)][1].copy()
            else:
                x0 = best_x + 1.0 * rng.standard_normal(d)
                x0 = np.clip(x0, -5, 5)

            sigma0 = 1.5 if restart == 0 else 0.8

            # Run CMA-ES
            cma_result_x, cma_result_f = cma_es(x0, sigma0, calls_made + budget_per_restart)

            # Refine with Nelder-Mead if we have budget left
            nm_budget = min(30, budget - calls_made)
            if nm_budget > 0 and time.time() < deadline:
                nm_result_x, nm_result_f = nelder_mead(cma_result_x, nm_budget, sigma=0.3)
                if nm_result_f < best_fx:
                    best_fx = nm_result_f
                    best_x = nm_result_x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
