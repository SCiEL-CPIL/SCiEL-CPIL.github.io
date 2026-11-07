"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution (DE) for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE Parameters
        # Population size: typically 5*dim to 10*dim. 15 is good for dim=5.
        npop = 15 
        # Scaling factor
        F = 0.8
        # Crossover rate
        CR = 0.9
        
        # Calculate generations
        # We need npop calls per generation
        max_generations = budget // npop
        
        # Initialize Population
        pop = rng.uniform(-5, 5, (npop, n))
        fitness = np.zeros(npop)
        
        calls_used = 0
        for i in range(npop):
            if calls_used >= budget:
                break
            fitness[i] = func(pop[i])
            calls_used += 1
            
        # Find initial best
        best_idx = np.argmin(fitness)
        best_f = fitness[best_idx]
        best_x = pop[best_idx].copy()

        # Main Loop
        for gen in range(max_generations):
            # Sort population by fitness to identify best for 'DE/best/1' if needed, 
            # but we will use 'DE/rand/1' which is more robust against local minima.
            # However, 'DE/current-to-best/1' can converge faster. 
            # Let's stick to standard 'DE/rand/1/bin' for robustness.
            
            for i in range(npop):
                if calls_used >= budget:
                    break
                
                # Select 3 distinct random individuals different from i
                # We need a, b, c != i and a != b != c
                indices = [j for j in range(npop) if j != i]
                a, b, c = rng.choice(indices, size=3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                mutant = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to bounds
                mutant = np.clip(mutant, -5, 5)
                
                # Crossover
                trial = pop[i].copy()
                r = rng.random()
                j_rand = rng.integers(0, n)
                
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Evaluate
                f_trial = func(trial)
                calls_used += 1
                
                # Selection
                if f_trial <= fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
