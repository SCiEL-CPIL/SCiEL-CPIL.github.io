"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE/rand/1/bin) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s/330s limit
        
        # Use a deterministic seed for reproducibility if needed, 
        # but the task implies the function itself might be stochastic or fixed per instance.
        # Using a fixed seed ensures the same behavior if the evaluator re-runs, 
        # but typically we want robustness. Let's use a seed derived from the function object 
        # to diversify if the evaluator calls it multiple times with different instances, 
        # although the prompt says "creates Optimizer... and calls it once" per instance.
        # Actually, the prompt says "For each instance in [1, 2, 3] the evaluator creates Optimizer...".
        # So we can just use a fixed seed or random. Let's use random to avoid bad luck with a specific seed.
        rng = np.random.default_rng()
        
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        N = 10  # Population size
        F = 0.8 # Mutation factor
        CR = 0.9 # Crossover rate
        
        # Ensure budget allows at least one full generation
        # If budget < N, we just do random search
        if budget < N:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            for _ in range(budget - 1):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f

        # Initialize population
        # Uniform in [-5, 5]
        population = rng.uniform(-5, 5, (N, dim))
        
        # Evaluate initial population
        fitness = np.empty(N)
        for i in range(N):
            if time.time() > deadline:
                break
            fitness[i] = func(population[i])
        
        calls_made = N
        best_idx = np.argmin(fitness)
        best_f = fitness[best_idx]
        best_x = population[best_idx].copy()
        
        # Main DE Loop
        # We have budget - N evaluations left.
        # Each generation consumes N evaluations.
        # generations = (budget - N) // N
        # But we must also check time.
        
        while calls_made + N <= budget:
            if time.time() > deadline:
                break
            
            # Perform one generation
            # Update population in place to save memory, but we need to keep old population for mutation
            # Actually, standard DE updates one individual at a time.
            
            for i in range(N):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                # Simple approach: pick random indices, ensure distinct
                r1, r2, r3 = i, i, i
                while r1 == i:
                    r1 = rng.integers(N)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(N)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(N)
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover (Binomial)
                # Pick a random index to ensure at least one gene is from v
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
