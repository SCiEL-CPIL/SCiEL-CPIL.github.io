"""Portfolio-based optimizer for BBOB f3 (Rastrigin) with DE and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Differential Evolution + Local Refinement optimizer.
    1. Brief exploratory random sampling.
    2. Differential Evolution (DE) for global search and basin hopping.
    3. (1+1)-ES local refinement for high-precision polishing.
    4. Time-bounded with explicit deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        def check_deadline():
            return time.time() > deadline

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling
        n_initial = min(50, budget // 10)
        for _ in range(n_initial):
            if check_deadline() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            safe_eval(x)

        # Phase 2: Differential Evolution (DE)
        # DE maintains a population and uses vector differences for exploration.
        # This is robust to multimodality (Rastrigin) and avoids the overhead of CMA-ES.
        if best_x is not None and calls_made < budget and not check_deadline():
            pop_size = 20
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate

            # Initialize population: include current best and random points
            population = [best_x.copy()]
            for _ in range(pop_size - 1):
                x = rng.uniform(-5, 5, dim)
                population.append(x)

            # Evaluate initial population
            for i in range(pop_size):
                if check_deadline() or calls_made >= budget:
                    break
                fx = safe_eval(population[i])
                population[i] = np.concatenate([population[i], [fx]])

            # DE loop
            max_generations = 20
            for gen in range(max_generations):
                if check_deadline() or calls_made >= budget:
                    break

                # Sort population by fitness
                population = sorted(population, key=lambda x: x[-1])

                # Update global best
                if population[0][-1] < best_f:
                    best_f = population[0][-1]
                    best_x = population[0][:-1].copy()

                # Check for convergence
                if population[-1][-1] - population[0][-1] < 1e-6:
                    break

                new_population = [population[0].copy()]  # Elitism: keep best

                for i in range(1, pop_size):
                    if check_deadline() or calls_made >= budget:
                        break

                    # Select three distinct random individuals
                    while True:
                        a = rng.integers(0, pop_size)
                        b = rng.integers(0, pop_size)
                        c = rng.integers(0, pop_size)
                        if a != b and b != c and c != a:
                            break

                    # Create trial vector
                    v = population[a][:-1] + F * (population[b][:-1] - population[c][:-1])
                    v = np.clip(v, -5, 5)

                    # Crossover
                    r = rng.uniform()
                    j_rand = rng.integers(0, dim)
                    trial_x = population[i][:-1].copy()
                    for j in range(dim):
                        if rng.uniform() < CR or j == j_rand:
                            trial_x[j] = v[j]

                    trial_x = np.clip(trial_x, -5, 5)
                    trial_f = safe_eval(trial_x)

                    # Selection
                    if trial_f <= population[i][-1]:
                        new_population.append(np.concatenate([trial_x, [trial_f]]))
                    else:
                        new_population.append(population[i].copy())

                population = new_population[:pop_size]

        # Phase 3: Local refinement using (1+1)-ES
        if best_x is not None and calls_made < budget and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            min_sigma = 1e-5
            max_sigma = 2.0
            successes = 0
            trials = 0

            # Limit local search budget
            local_budget = min(100, budget - calls_made)
            for _ in range(local_budget):
                if check_deadline() or calls_made >= budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_eval(y)
                trials += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.1
                    sigma = min(sigma, max_sigma)
                else:
                    sigma *= 0.9
                    sigma = max(sigma, min_sigma)

                # Adapt sigma based on success rate
                if trials % 15 == 0:
                    success_rate = successes / 15.0
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

                if sigma < min_sigma * 2:
                    break

        # Phase 4: Fine-tuning around the best solution
        if best_x is not None and calls_made < budget and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            min_sigma = 1e-6
            max_sigma = 0.5
            successes = 0
            trials = 0

            fine_budget = min(50, budget - calls_made)
            for _ in range(fine_budget):
                if check_deadline() or calls_made >= budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_eval(y)
                trials += 1

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    sigma *= 1.1
                    sigma = min(sigma, max_sigma)
                else:
                    sigma *= 0.9
                    sigma = max(sigma, min_sigma)

                if trials % 10 == 0:
                    success_rate = successes / 10.0
                    if success_rate > 0.4:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    sigma = np.clip(sigma, min_sigma, max_sigma)
                    successes = 0

                if sigma < min_sigma * 2:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
