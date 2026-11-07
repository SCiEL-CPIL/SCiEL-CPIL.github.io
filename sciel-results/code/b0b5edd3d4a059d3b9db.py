import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Differential Evolution Parameters
        # Larger population for better global coverage in 5D
        NP = 15
        F = 0.8
        CR = 0.9
        
        # 1. Initialization
        population = np.zeros((NP, dim))
        fitness = np.zeros(NP)
        
        for i in range(NP):
            x = rng.uniform(-4.0, 4.0, size=dim)
            population[i] = x
            fitness[i] = eval_func(x)
            
        idx_best = np.argmin(fitness)
        current_best_f = fitness[idx_best]
        current_best_x = population[idx_best].copy()
        
        # Stagnation counter
        stagnation_counter = 0
        last_best_f = current_best_f
        stagnation_limit = 5  # Generations without improvement before injecting randomness

        # 2. Main Loop
        while calls < budget - 1 and time_ok():
            # Check for stagnation
            if current_best_f < last_best_f - 1e-8:
                stagnation_counter = 0
                last_best_f = current_best_f
            else:
                stagnation_counter += 1

            # If stagnating, replace the worst individuals with random ones (Parallel Tempering-like exploration)
            if stagnation_counter >= stagnation_limit:
                # Replace the worst 2 individuals
                worst_indices = np.argsort(fitness)[-2:]
                for idx in worst_indices:
                    population[idx] = rng.uniform(-4.0, 4.0, size=dim)
                    fitness[idx] = eval_func(population[idx])
                    if fitness[idx] < current_best_f:
                        current_best_f = fitness[idx]
                        current_best_x = population[idx].copy()
                stagnation_counter = 0
                last_best_f = current_best_f

            # DE Step
            for i in range(NP):
                if calls >= budget or not time_ok():
                    break
                
                # Select 3 distinct random indices from [0, NP) excluding i
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                # Mutation: DE/rand/1
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Clip to search space
                v = np.clip(v, -5.0, 5.0)
                
                # Crossover: Binomial
                trial = population[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                        
                # Evaluate and Select
                f_trial = eval_func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < current_best_f:
                        current_best_f = f_trial
                        current_best_x = trial.copy()

        # Final check to ensure best_x is the absolute best found
        # (eval_func already tracks this, but just in case)
        
        return best_x, best_f
