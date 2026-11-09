"""MODEL-AND-OPTIMIZE approach: Halton exploration + CMA-ES + Nelder-Mead polish."""
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
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = population[0].copy()
                if fvals[0] < best_in_run:
                    best_in_run = fvals[0]

                # Recombination
                x_new = np.sum(weights[:, None] * population[:mu], axis=0)

                # Update evolution paths
                y = (x_new - x) / (sigma + 1e-30)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y

                # Adapt sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update covariance matrix
                delta = 0 if np.dot(p_sigma, p_sigma) / (1 - (1 - c_sigma) ** (2 * (eval_count // lam + 1))) < (1 + 2 / n) * chi_n ** 2 else 1
                rank_1 = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (population[i] - x) / (sigma + 1e-30)
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c_1 - c_mu + delta * c_1 * c_c * (2 - c_c)) * C + c_1 * rank_1 + c_mu * rank_mu

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                B = eigvecs
                D = np.sqrt(eigvals)

                # Update mean
                x = x_new

                # Check for stagnation
                if np.max(fvals[:mu]) - np.min(fvals[:mu]) < 1e-10 * (1 + np.abs(fvals[0])):
                    stagnation += 1
                else:
                    stagnation = 0

                if stagnation > 15:
                    break

            return x, best_in_run

        # Phase 1: Halton space-filling for systematic early exploration
        # Use enough points to get good AOCC early, but not too many to waste budget
        n_halton = min(60, max(30, budget // 8))

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
            if calls_made >= budget - 20:
                break
            x = halton_pts[i] * 10.0 - 5.0  # map [0,1] to [-5,5]
            x = np.clip(x, -5.0, 5.0)
            fx = eval_func(x)
            if fx is None:
                break

        if best_x is None:
            return best_x, best_f

        # Phase 2: CMA-ES from best point found by Halton
        # CMA-ES is better for multi-modal landscapes than (1+1)-ES
        if calls_made < budget - 30:
            remaining_cma = budget - calls_made - 30
            # Start with moderate sigma to explore around the best point
            cma_es(best_x.copy(), sigma0=1.0, max_evals=remaining_cma, seed=100)

        # Phase 3: Additional CMA-ES restart from a perturbed point if budget remains
        if best_x is not None and calls_made < budget - 20:
            remaining_restart = budget - calls_made - 20
            if remaining_restart > 10:
                # Perturb from best with smaller step
                rng_restart = np.random.default_rng(200)
                x_start = best_x + 0.5 * rng_restart.standard_normal(dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                cma_es(x_start, sigma0=0.5, max_evals=remaining_restart, seed=200)

        # Phase 4: Final Nelder-Mead polish from best point
        if best_x is not None and calls_made < budget - 5:
            remaining_polish = budget - calls_made - 5
            if remaining_polish > 10:
                nelder_mead(best_x.copy(), max_evals=remaining_polish, seed=5000)

        # Phase 5: Very fine local search if budget remains
        if best_x is not None and calls_made < budget - 2:
            remaining_fine = budget - calls_made - 2
            if remaining_fine > 5:
                # Tiny step CMA-ES for final precision
                cma_es(best_x.copy(), sigma0=0.01, max_evals=remaining_fine, seed=6000)

        return best_x, best_f
# EVOLVE-BLOCK-END
