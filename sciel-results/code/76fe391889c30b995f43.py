"""MODEL-AND-OPTIMIZE approach: Halton exploration + CMA-ES with adaptive restarts + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton space-filling + CMA-ES with adaptive restarts + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim, base_start=2):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = base_start + d
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        idx, rem = divmod(idx, base)
                        f /= base
                        r += rem * f
                    points[i, d] = r
            return points

        def nelder_mead(x0, max_evals, seed=42):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_x, best_f, calls_made
            n = dim

            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += 0.25 * max(0.1, abs(x0[i]))

            # Evaluate simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fx = eval_func(simplex[i])
                if fx is None:
                    return
                fvals[i] = fx

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for iteration in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = eval_func(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = eval_func(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < fvals[-1]:
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = eval_func(x_contract)
                    if f_contract is None:
                        break
                    if f_contract < min(f_reflect, fvals[-1]):
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fx = eval_func(simplex[i])
                            if fx is None:
                                return
                            fvals[i] = fx
                        break

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + np.abs(fvals[0])):
                    break

            return simplex[0], fvals[0]

        def cma_es(x0, sigma0, max_evals, seed=42):
            """CMA-ES implementation with adaptive step size."""
            nonlocal best_x, best_f, calls_made
            rng = np.random.default_rng(seed)
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of best points for recombination

            # Initialize
            x = np.asarray(x0, dtype=np.float64).copy()
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)

            # Weights for weighted recombination
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)

            # State variables
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (20 * n ** 2))

            # Evaluate initial point
            fx = eval_func(x)
            if fx is None:
                return x, fx

            # Main loop
            eval_count = 0
            stagnation = 0
            best_in_run = fx

            while eval_count < max_evals:
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                # Generate population
                population = np.zeros((lam, n))
                fvals = np.zeros(lam)

                for i in range(lam):
                    if time.time() > deadline:
                        break
                    if calls_made >= budget - 2:
                        break

                    y = rng.standard_normal(n)
                    z = B @ (D * y)
                    x_i = x + sigma * z
                    x_i = np.clip(x_i, -5.0, 5.0)

                    f_i = eval_func(x_i)
                    if f_i is None:
                        break
                    population[i] = x_i
                    fvals[i] = f_i
                    eval_count += 1

                # Check if all evaluations succeeded
                if any(np.isnan(fvals)):
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                population = population[idx]
                fvals = fvals[idx]

                # Update best
                if fvals[0] < best_in_run:
                    best_in_run = fvals[0]
                    stagnation = 0
                else:
                    stagnation += 1

                # Weighted recombination
                x_new = np.sum(weights[:, np.newaxis] * population[:mu], axis=0)

                # Update covariance matrix
                # Rank-one update
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_new - x) / sigma
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) > (1.4 + 2 / (n + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Rank-mu update
                delta = (x_new - x) / sigma
                C = (1 + c_1 * (1 - 1 - c_1) + c_mu) * C
                C += c_1 * (np.outer(delta, delta) - C)
                C += c_mu * np.sum(weights[:mu, np.newaxis, np.newaxis] * np.outer(population[:mu] - x, population[:mu] - x) / (sigma ** 2), axis=0)

                # Eigen-decomposition
                evals, evecs = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-20)
                D = np.sqrt(evals)
                B = evecs
                C = B @ np.diag(D ** 2) @ B.T

                # Update mean
                x = x_new

                # Update p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + np.abs(fvals[0])):
                    break

                # Check stagnation
                if stagnation > 15:
                    break

            return x, best_in_run

        # Phase 1: Halton space-filling for systematic early exploration
        n_halton = min(80, max(40, budget // 6))

        # Add center point first
        x0 = np.zeros(dim)
        fx0 = eval_func(x0)
        if fx0 is None:
            return best_x, best_f

        # Halton points mapped to [-5, 5]^dim
        halton_pts = halton_sequence(n_halton - 1, dim)
        for i in range(n_halton - 1):
            if time.time() > deadline:
                break
            if calls_made >= budget - 10:
                break
            x = halton_pts[i] * 10.0 - 5.0  # map [0,1] to [-5,5]
            x = np.clip(x, -5.0, 5.0)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: CMA-ES with adaptive restarts
        remaining = budget - calls_made
        if remaining < 20:
            return best_x, best_f

        # Reserve some for final polish
        polish_budget = min(50, remaining // 5)
        cma_budget_total = remaining - polish_budget

        # Run CMA-ES from best point with adaptive restarts
        best_cma_f = best_f
        best_cma_x = best_x.copy()
        num_restarts = 0
        max_restarts = 3

        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget - polish_budget - 5:
                break

            # Determine initial sigma based on restart number
            if restart == 0:
                sigma_init = 1.0
            else:
                sigma_init = 0.5 * (restart + 1)

            # Run CMA-ES
            x_end, f_end = cma_es(best_cma_x, sigma_init, cma_budget_total // (max_restarts - restart), seed=1000 + restart * 100)

            if f_end < best_cma_f:
                best_cma_f = f_end
                best_cma_x = x_end.copy()

            # Update global best
            if f_end < best_f:
                best_f = f_end
                best_x = x_end.copy()

            # Check if we should continue
            if calls_made >= budget - polish_budget - 5:
                break

            # If no improvement, break
            if f_end >= best_cma_f - 1e-6:
                break

            num_restarts += 1

        # Phase 3: Final Nelder-Mead polish from best point
        if best_x is not None and calls_made < budget - 5:
            remaining_polish = budget - calls_made - 5
            if remaining_polish > 10:
                nelder_mead(best_x.copy(), max_evals=remaining_polish, seed=5000)

        # Phase 4: Very fine local search if budget remains
        if best_x is not None and calls_made < budget - 2:
            remaining_fine = budget - calls_made - 2
            if remaining_fine > 5:
                # Tiny step CMA-ES for final precision
                cma_es(best_x.copy(), sigma0=0.01, max_evals=remaining_fine, seed=6000)

        return best_x, best_f
# EVOLVE-BLOCK-END
