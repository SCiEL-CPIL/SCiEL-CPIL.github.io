"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based DE with CMA-ES refinement for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution found
        best_x = None
        best_f = np.inf
        total_calls = 0

        def safe_eval(x):
            nonlocal total_calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            total_calls += 1
            return func(x)

        def de_search(x0, F, CR, pop_size, max_iter, deadline, remaining_calls):
            """Differential Evolution rand/1/bin with population diversity."""
            nonlocal best_x, best_f, total_calls

            # Initialize population
            pop = []
            for i in range(pop_size):
                if remaining_calls <= 0 or time.time() > deadline:
                    break
                if i == 0:
                    x = x0.copy()
                else:
                    x = x0 + rng.normal(0, 1.5, n)
                x = np.clip(x, -5.0, 5.0)
                f = safe_eval(x)
                pop.append((f, x.copy()))
                remaining_calls -= 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            if len(pop) < 3:
                return best_x, best_f, remaining_calls

            # Sort population
            pop.sort(key=lambda t: t[0])

            for it in range(max_iter):
                if time.time() > deadline or remaining_calls < pop_size:
                    break

                new_pop = []
                # Keep the current best individuals (elitism)
                for i in range(min(3, len(pop))):
                    if remaining_calls <= 0:
                        break
                    new_pop.append(pop[i])

                # Generate new candidates via DE/rand/1/bin
                for i in range(pop_size - min(3, len(pop))):
                    if remaining_calls <= 0 or time.time() > deadline:
                        break
                    # Pick three distinct random individuals
                    idxs = rng.choice(len(pop), size=3, replace=False)
                    a, b, c = pop[idxs[0]][1], pop[idxs[1]][1], pop[idxs[2]][1]
                    # Mutation: v = a + F * (b - c)
                    v = a + F * (b - c)
                    # Crossover: binomial
                    trial = v.copy()
                    jrand = rng.integers(0, n)
                    for j in range(n):
                        if rng.random() < CR or j == jrand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop[i % len(pop)][1][j]
                    trial = np.clip(trial, -5.0, 5.0)
                    ft = safe_eval(trial)
                    remaining_calls -= 1
                    # Selection
                    if ft < pop[i % len(pop)][0]:
                        new_pop.append((ft, trial))
                    else:
                        new_pop.append(pop[i % len(pop)])

                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()

                # Sort and keep best pop_size
                new_pop.sort(key=lambda t: t[0])
                pop = new_pop[:pop_size]

            return best_x, best_f, remaining_calls

        def cma_es_refine(x0, sigma0, max_iter, deadline, remaining_calls):
            """CMA-ES for local refinement."""
            nonlocal best_x, best_f, total_calls

            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)

            # Evaluate initial point
            fx = safe_eval(x)
            remaining_calls -= 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            for it in range(max_iter):
                if time.time() > deadline or remaining_calls < lambda_:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    if remaining_calls <= 0 or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_eval(y)
                    remaining_calls -= 1
                    offspring.append((fy, y))
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])

                # Weighted mean
                xnew = np.sum([weights[i] * offspring[i][1] for i in range(mu)], axis=0)

                # Step size control
                ydiff = (xnew - x) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D * ydiff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) < \
                       (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                hs = float(hsig)
                pc = (1 - cc) * pc + hs * cc / D * ydiff

                # Covariance matrix adaptation
                rank1 = np.outer(pc, pc)
                delta_h = (1 - hs) * cc * (1 - cc)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - x) / sigma
                    rankmu += np.outer(weights[i] * zi, weights[i] * zi)
                C = (1 + c1 * delta_h - c1) * C + c1 * rank1 + cmu * rankmu

                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                x = xnew
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ydiff) / np.sqrt(2 * n) - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

            return best_x, best_f, remaining_calls

        # Phase 1: Population-based DE for global exploration
        # Use ~55% of budget for DE
        de_budget = int(budget * 0.55)
        pop_size = 20
        F = 0.8
        CR = 0.7

        # Multiple DE restarts with diverse initializations
        n_de_restarts = 4
        de_calls_per_restart = de_budget // n_de_restarts

        for r in range(n_de_restarts):
            if time.time() > deadline or total_calls >= budget:
                break

            remaining_calls = budget - total_calls
            if remaining_calls < pop_size * 2:
                break

            # Different starting points for each restart
            if r == 0:
                x0 = np.zeros(n)
            elif r == 1:
                x0 = rng.uniform(-3, 3, n)
            elif r == 2:
                x0 = rng.uniform(-4, 4, n)
            else:
                x0 = rng.uniform(-5, 5, n)

            max_iter = max(1, de_calls_per_restart // pop_size)
            _, _, remaining_calls = de_search(x0, F, CR, pop_size, max_iter, deadline, remaining_calls)

        # Phase 2: CMA-ES refinement around the best solution
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining_calls = budget - total_calls
            # Use remaining budget for CMA-ES refinement
            cma_budget = min(remaining_calls, 150)
            max_iter_cma = max(1, cma_budget // 10)
            sigma0 = 0.5  # Start with moderate sigma for refinement
            _, _, _ = cma_es_refine(best_x, sigma0, max_iter_cma, deadline, remaining_calls)

        # Phase 3: Final local polish with small perturbations
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining_calls = budget - total_calls
            polish_calls = min(30, remaining_calls)
            local_sigma = 0.05
            for _ in range(polish_calls):
                if time.time() > deadline or total_calls >= budget:
                    break
                y = best_x + local_sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_eval(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y
                local_sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
