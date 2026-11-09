import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Differential Evolution for global exploration
        # DE is good at finding the right basin in multimodal landscapes
        if not check_deadline():
            pop_size = 20
            # Initialize population uniformly in the box
            population = rng.uniform(-5, 5, (pop_size, dim))
            fitnesses = np.array([evaluate(ind) for ind in population])
            
            # Track best
            best_idx = np.argmin(fitnesses)
            best_x = population[best_idx].copy()
            best_f = fitnesses[best_idx]
            
            # DE parameters
            F = 0.8  # Mutation strength
            CR = 0.9  # Crossover rate
            
            # Run DE for ~300 calls
            de_budget = int(total_budget * 0.6)
            de_calls = pop_size
            while not check_deadline() and de_calls < de_budget:
                # Create new population
                new_population = []
                for i in range(pop_size):
                    if check_deadline() or de_calls >= de_budget:
                        break
                    # Mutation
                    # Pick 3 distinct indices
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    v = population[a] + F * (population[b] - population[c])
                    # Crossover
                    trial = rng.choice([0, 1], dim)
                    j_rand = rng.integers(dim)
                    trial[j_rand] = 1
                    u = np.where(trial == 1, v, population[i])
                    u = np.clip(u, -5, 5)
                    f_u = evaluate(u)
                    de_calls += 1
                    # Selection
                    if f_u <= fitnesses[i]:
                        new_population.append(u)
                        fitnesses[i] = f_u
                    else:
                        new_population.append(population[i])
                
                if len(new_population) < pop_size:
                    break
                population = np.array(new_population)
                
                # Update global best
                idx = np.argmin(fitnesses)
                if fitnesses[idx] < best_f:
                    best_f = fitnesses[idx]
                    best_x = population[idx].copy()

        # Phase 2: CMA-ES for local refinement
        if not check_deadline() and best_x is not None:
            # Initialize CMA-ES around the best point found by DE
            x_mean = best_x.copy()
            sigma = 1.0 # Start with a smaller step size for refinement
            
            lambda_pop = 10
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

            cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
            cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
            c1 = 2 / ((lambda_pop + 2) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2) ** 2 + mu_eff))

            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Evaluate mean to ensure it's in the population
            f_mean = evaluate(x_mean)
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()

            while not check_deadline():
                offspring = []
                for i in range(lambda_pop):
                    if check_deadline():
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = evaluate(x_new)
                    offspring.append((x_new, f_new))

                if len(offspring) == 0:
                    break

                offspring.sort(key=lambda x: x[1])

                # Update mean
                x_new_mean = np.sum([weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (B.T @ (x_new_mean - x_mean))))
                hsig = (np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(lambda_pop + 1 - mu_eff))
                if hsig:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))

                # Update covariance
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_new_mean - x_mean) / sigma
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)

                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                x_mean = x_new_mean

                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-12))
                except:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-12))

                # Check convergence
                if np.linalg.norm(D) < 1e-10 or sigma < 1e-10:
                    break

        return best_x, best_f
