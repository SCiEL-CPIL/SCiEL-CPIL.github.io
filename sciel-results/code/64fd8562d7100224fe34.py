"""Differential Evolution for BBOB f4 (Büche-Rastrigin) on [-5,5]^5."""
import numpy as np
import time

class Optimizer:
    """Pure Differential Evolution with static population for global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Population size: large enough for diversity, small enough to fit in budget
        # With 500 calls and pop=50, we get ~10 generations
        pop_size = 50
        F = 0.8  # Mutation factor
        CR = 0.5  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]^5
        population = rng.uniform(-5, 5, (pop_size, self.dim))
        
        # Evaluate initial population
        fitness = np.empty(pop_size)
        calls = 0
        for i in range(pop_size):
            if time.time() - start_time > time_limit:
                break
            fitness[i] = func(population[i])
            calls += 1
        
        if calls < pop_size:
            # Not enough budget or time for full initialization
            # Just return the best found
            best_idx = np.argmin(fitness[:calls])
            return population[best_idx], fitness[best_idx]
        
        # Track global best
        x_best = population[np.argmin(fitness)].copy()
        f_best = fitness.min()
        
        # DE loop
        while calls < self.budget:
            if time.time() - start_time > time_limit:
                break
            
            for i in range(pop_size):
                if calls >= self.budget:
                    break
                if time.time() - start_time > time_limit:
                    break
                
                # Select three distinct random individuals different from i
                while True:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    if a != i and b != i and c != i:
                        break
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = population[a] + F * (population[b] - population[c])
                v = np.clip(v, -5, 5)
                
                # Crossover: uniform crossover between x_i and v
                trial = population[i].copy()
                r = rng.random(self.dim)
                j_rand = rng.integers(0, self.dim)
                trial[r > CR] = v[r > CR]
                trial[j_rand] = v[j_rand]  # Ensure at least one gene from v
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                f_trial = func(trial)
                calls += 1
                
                # Selection: keep the better of trial and current
                if f_trial <= fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    
                    if f_trial < f_best:
                        x_best = trial.copy()
                        f_best = f_trial
        
        return x_best, f_best
