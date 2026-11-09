"""Optimizer for BBOB f17 (Schaffer's F7) using island-based CMA-ES with Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-based CMA-ES with Nelder-Mead refinement for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 330s limit, leave 45s safety margin

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

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for i in range(d):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                x = np.zeros(d)
                x[i] = val
                points.append(x.copy())
        for _ in range(10):
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

        def nelder_mead(x0, max_calls, max_iter=50):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_x, best_fx, calls_made

            n = d
            delta = 0.3
            simplex = [x0.copy()]
            for i in range(n):
                pt = x0.copy()
                pt[i] += delta
                simplex.append(np.clip(pt, -5, 5))

            fvals = []
            for pt in simplex:
                if not time_ok() or calls_made >= max_calls:
                    return x0, np.inf
                fvals.append(call_func(pt))

            local_best_idx = int(np.argmin(fvals))
            local_best_fx = fvals[local_best_idx]
            local_best_x = simplex[local_best_idx].copy()

            if local_best_fx < best_fx:
                best_fx = local_best_fx
                best_x = local_best_x.copy()

            alpha, gamma, rho = 1.0, 2.0, 0.5

            for _ in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in order]
                fvals = [fvals[i] for i in order]

                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst_pt = simplex[-1]

                # Reflection
                reflected = centroid + alpha * (centroid - worst_pt)
                reflected = np.clip(reflected, -5, 5)
                if not time_ok() or calls_made >= max_calls:
                    break
                ref_fx = call_func(reflected)

                if ref_fx < fvals[0]:
                    # Expansion
                    expanded = centroid + gamma * (reflected - centroid)
                    expanded = np.clip(expanded, -5, 5)
                    if not time_ok() or calls_made >= max_calls:
                        break
                    exp_fx = call_func(expanded)
                    if exp_fx < ref_fx:
                        simplex[-1] = expanded
                        fvals[-1] = exp_fx
                    else:
                        simplex[-1] = reflected
                        fvals[-1] = ref_fx
                elif ref_fx < fvals[-2]:
                    simplex[-1] = reflected
                    fvals[-1] = ref_fx
                else:
                    # Contraction
                    contracted = centroid + rho * (worst_pt - centroid)
                    contracted = np.clip(contracted, -5, 5)
                    if not time_ok() or calls_made >= max_calls:
                        break
                    con_fx = call_func(contracted)
                    if con_fx < fvals[-1]:
                        simplex[-1] = contracted
                        fvals[-1] = con_fx
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not time_ok() or calls_made >= max_calls:
                                break
                            fvals[i] = call_func(simplex[i])
                        if not time_ok() or calls_made >= max_calls:
                            break

                lb_idx = int(np.argmin(fvals))
                if fvals[lb_idx] < local_best_fx:
                    local_best_fx = fvals[lb_idx]
                    local_best_x = simplex[lb_idx].copy()
                    if local_best_fx < best_fx:
                        best_fx = local_best_fx
                        best_x = local_best_x.copy()

            return local_best_x, local_best_fx

        def cma_es_island(x0, sigma0, max_calls, seed_offset=0):
            """Single CMA-ES run for one island."""
            nonlocal best_x, best_fx, calls_made

            n = d
            pop_size = max(4, int(4 + 3 * np.log(n)))
            mu = pop_size // 2

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size control
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)

            # Covariance matrix adaptation
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1]) / ((n + 2) ** 2 + mu_eff))

            # State
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            def C_eig_decomp(C):
                """Eigen decomposition with caching for stability."""
                D, V = np.linalg.eigh(C)
                D = np.clip(D, 1e-20, None)
                return D, V

            def C_sqrt_update(C, D, V):
                """Update C^(1/2) using eigen decomposition."""
                D_sqrt = np.sqrt(D)
                return V @ np.diag(D_sqrt) @ V.T

            # Evaluate initial population
            pop = []
            fvals = []
            for i in range(pop_size):
                if not time_ok() or calls_made >= max_calls:
                    return mean, np.inf
                z = rng.standard_normal(n)
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)
                fvals.append(call_func(x))

            # Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()

            # CMA-ES iterations
            max_iter = max(1, (max_calls - calls_made) // pop_size)
            for it in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break

                # Update mean
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (pop[i] - mean) / sigma
                mean = mean + sigma * y_w
                mean = np.clip(mean, -5, 5)

                # Update step size
                y_sigma = np.zeros(n)
                for i in range(mu):
                    y_sigma += weights[i] * (pop[i] - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update covariance
                y_c = np.zeros(n)
                for i in range(mu):
                    y_c += weights[i] * (pop[i] - mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c

                C = ((1 - c_1 - c_mu) * C +
                     c_1 * (2 - c_c) * C +
                     c_1 * (np.outer(p_c, p_c) - (1 - c_c) * C) +
                     c_mu * np.sum([weights[i] * np.outer(pop[i] - mean, pop[i] - mean)
                                    for i in range(mu)], axis=0) / sigma ** 2)

                # Symmetrize
                C = 0.5 * (C + C.T)

                # Ensure positive definite
                D, V = C_eig_decomp(C)
                C = V @ np.diag(D) @ V.T
                C_sqrt = C_sqrt_update(C, D, V)

                # Generate new population
                new_pop = []
                new_fvals = []
                for i in range(pop_size):
                    if not time_ok() or calls_made >= max_calls:
                        break
                    z = rng.standard_normal(n)
                    x = mean + sigma * (C_sqrt @ z)
                    x = np.clip(x, -5, 5)
                    new_pop.append(x)
                    new_fvals.append(call_func(x))

                if len(new_fvals) < pop_size:
                    break

                order = np.argsort(new_fvals)
                pop = [new_pop[i] for i in order]
                fvals = [new_fvals[i] for i in order]

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = pop[0].copy()

            return mean, fvals[0] if fvals else np.inf

        # Phase 2: Island-based CMA-ES with Nelder-Mead refinement
        num_islands = 3
        island_budgets = [150, 150, 100]  # Total ~400, leaving room for NM

        for island_idx in range(num_islands):
            if not time_ok():
                break

            if island_idx == 0:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                # Perturb from best for diversity
                x0 = best_x + rng.normal(0, 1.5, d)
                x0 = np.clip(x0, -5, 5)
                sigma0 = 0.8

            max_calls_for_island = calls_made + island_budgets[island_idx]
            _, _ = cma_es_island(x0, sigma0, max_calls_for_island, seed_offset=island_idx)

        # Phase 3: Nelder-Mead refinement on best point
        if time_ok() and calls_made < budget:
            nm_calls = min(50, budget - calls_made)
            if nm_calls > 10:
                x0 = best_x.copy()
                _, _ = nelder_mead(x0, calls_made + nm_calls)

        # Phase 4: Final local polish
        if time_ok() and calls_made < budget:
            polish_calls = min(10, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.05 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
