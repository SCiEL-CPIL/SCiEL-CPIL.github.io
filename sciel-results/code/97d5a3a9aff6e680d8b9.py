"""Island-model DE + CMA-ES hybrid for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Island DE for basin exploration + CMA-ES/L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Structured grid for early AOCC ---
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = []
        for i1 in grid_vals:
            for i2 in grid_vals:
                for i3 in grid_vals:
                    for i4 in grid_vals:
                        for i5 in grid_vals:
                            grid_points.append(np.array([i1, i2, i3, i4, i5]))
        rng.shuffle(grid_points)
        grid_budget = min(len(grid_points), max(20, budget // 5))
        for pt in grid_points[:grid_budget]:
            if eval_count >= budget - 50:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # --- Phase 2: Island-model Differential Evolution ---
        # Multiple islands to explore different basins of the 21-peak landscape
        n_islands = 4
        pop_per_island = 12
        de_budget = min(budget - eval_count - 120, 200)
        n_de_gens = de_budget // (n_islands * pop_per_island) if de_budget > 0 else 0

        if n_de_gens > 0 and time.time() < deadline:
            # Initialize islands with diverse starting points
            islands = []
            for k in range(n_islands):
                center = rng.uniform(-4.0, 4.0, dim)
                pop = center + rng.normal(0, 1.5, (pop_per_island, dim))
                pop = np.clip(pop, -5.0, 5.0)
                # Evaluate initial population
                fitness = np.empty(pop_per_island)
                for i in range(pop_per_island):
                    if eval_count >= budget - 120:
                        fitness[i] = float('inf')
                        continue
                    if time.time() > deadline:
                        fitness[i] = float('inf')
                        continue
                    fitness[i] = wrapped_func(pop[i])
                    update_best(pop[i], fitness[i])
                islands.append({'pop': pop, 'fit': fitness})

            # DE parameters
            F = 0.8  # differential weight
            CR = 0.9  # crossover rate

            for gen in range(n_de_gens):
                if eval_count >= budget - 120 or time.time() > deadline:
                    break

                for k in range(n_islands):
                    if eval_count >= budget - 120 or time.time() > deadline:
                        break
                    pop = islands[k]['pop']
                    fit = islands[k]['fit']

                    for i in range(pop_per_island):
                        if eval_count >= budget - 120 or time.time() > deadline:
                            break
                        # Select three distinct random individuals
                        while True:
                            a, b, c = rng.choice(pop_per_island, 3, replace=False)
                            if len({a, b, c, i}) == 4:
                                break

                        # DE/rand/1
                        v = pop[a] + F * (pop[b] - pop[c])
                        v = np.clip(v, -5.0, 5.0)

                        # Binary crossover
                        j_rand = rng.integers(dim)
                        trial = pop[i].copy()
                        for j in range(dim):
                            if rng.random() < CR or j == j_rand:
                                trial[j] = v[j]
                        trial = np.clip(trial, -5.0, 5.0)

                        # Evaluate trial
                        f_trial = wrapped_func(trial)
                        update_best(trial, f_trial)

                        # Selection
                        if f_trial < fit[i]:
                            pop[i] = trial
                            fit[i] = f_trial

        # --- Phase 3: CMA-ES from best point ---
        def run_cmaes(start_x, sigma0, max_evals):
            nonlocal eval_count, deadline
            n = dim
            lambda_ = int(4 + 3 * n)
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            mean = start_x.copy()
            sigma = sigma0
            C = np.eye(n)
            ps = np.zeros(n)
            prev_mean = mean.copy()
            local_best_f = best_f
            local_best_x = best_x.copy()
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            while eval_count < max_evals and time.time() < deadline:
                if time.time() > deadline - 5:
                    break
                population = []
                fitness = []
                for i in range(lambda_):
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    try:
                        B = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        B = np.eye(n)
                    x_i = mean + sigma * (B @ z)
                    x_i = np.clip(x_i, -5.0, 5.0)
                    f_i = wrapped_func(x_i)
                    if f_i < local_best_f:
                        local_best_f = f_i
                        local_best_x = x_i.copy()
                    population.append(x_i)
                    fitness.append(f_i)

                if len(population) < mu:
                    break

                idx = np.argsort(fitness)
                population = [population[i] for i in idx]
                fitness = [fitness[i] for i in idx]

                new_mean = sum(weights[i] * population[i] for i in range(mu))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (new_mean - prev_mean) / sigma
                diff = (new_mean - prev_mean) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(diff, diff) + (1 - (1 - cs) ** 2) * C)
                for i in range(mu):
                    xi = (population[i] - prev_mean) / sigma
                    C += cmu * weights[i] * np.outer(xi, xi)
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(cs / np.sqrt(n) * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                prev_mean = mean
                mean = new_mean
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-6 * np.eye(n)

            return local_best_x, local_best_f

        cmaes_budget = min(150, budget - eval_count - 40)
        if cmaes_budget > 20 and time.time() < deadline:
            cmaes_x, cmaes_f = run_cmaes(best_x.copy(), 1.0, eval_count + cmaes_budget)
            if cmaes_f < best_f:
                best_f = cmaes_f
                best_x = cmaes_x

        # --- Phase 4: Multi-start L-BFGS-B refinement ---
        start_points = [best_x.copy()]
        for scale in [1.0, 0.5, 0.25]:
            for _ in range(2):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)
        for _ in range(4):
            sp = rng.uniform(-4.5, 4.5, dim)
            start_points.append(sp)

        unique_starts = []
        for sp in start_points:
            is_dup = any(np.linalg.norm(sp - us) < 0.1 for us in unique_starts)
            if not is_dup:
                unique_starts.append(sp)

        for sp in unique_starts:
            if eval_count >= budget - 10 or time.time() > deadline:
                break
            try:
                result = minimize(
                    wrapped_func, sp, method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # --- Phase 5: Final polish ---
        if eval_count < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func, best_x, method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        if eval_count < budget - 3 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func, best_x, method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 40, 'maxls': 25, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Fill remaining budget
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.005, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
