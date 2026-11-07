"""New optimizer using Differential Evolution."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Differential Evolution optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # Popsize typically 5*dim to 10*dim. For dim=5, 25 is good.
        # But budget is 500. 500 / 25 = 20 generations.
        # Let's try popsize=10. 500/10 = 50 generations.
        popsize = 10
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, (popsize, dim))
        
        # Evaluate initial population
        fitness = np.zeros(popsize)
        calls = 0
        for i in range(popsize):
            fitness[i] = func(pop[i])
            calls += 1
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # Main loop
        while calls < budget:
            # Determine how many more generations we can afford
            # Each generation costs popsize evaluations
            if calls + popsize > budget:
                # If we can't do a full generation, we might do a partial one or stop.
                # For simplicity and strict budget adherence, stop if we can't do a full generation?
                # Or do partial. Let's do partial to use all budget.
                remaining_calls = budget - calls
                # We will evaluate 'remaining_calls' individuals from the next generation
                # But DE needs the whole population to update. 
                # It's safer to just stop if we can't complete a generation, 
                # or implement a "partial generation" logic which is tricky.
                # Given 500 % 10 == 0, we can do exactly 50 generations.
                # Initial 10 + 50*10 = 510? No.
                # Init: 10 calls.
                # Loop:
                # Gen 1: 10 calls. Total 20.
                # ...
                # Gen 49: 10 calls. Total 10 + 490 = 500.
                # So we can do exactly 49 generations after init?
                # Let's check:
                # Calls used: 10 (init)
                # Budget left: 490
                # Generations: 490 / 10 = 49.
                # So we do 49 generations.
                
                break
            
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            for i in range(popsize):
                if calls >= budget:
                    break
                
                # Select 3 distinct random individuals different from i
                r1, r2, r3 = rng.choice(popsize, 3, replace=False)
                # Ensure they are not i? 
                # rng.choice without replacement might pick i. 
                # Standard DE requires distinct from i.
                while r1 == i or r2 == i or r3 == i or r1 == r2 or r2 == r3 or r1 == r3:
                    r1, r2, r3 = rng.choice(popsize, 3, replace=False)
                
                # Create trial vector
                # v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to box
                v = np.clip(v, -5, 5)
                
                # Crossover
                # Binary crossover: choose a random index j_rand
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate trial
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            pop = new_pop
            fitness = new_fitness
        
        return best_x, best_f
# EVOLVE-BLOCK-END
