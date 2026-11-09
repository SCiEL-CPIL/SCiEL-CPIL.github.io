# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # DE Parameters
        NP = 15  # Population size, ~3*dim
        F = 0.7  # Mutation factor
        CR = 0.5  # Crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, size=(NP, dim))
        fit = np.empty(NP)
        calls_made = 0
        
        for i in range(NP):
            fit[i] = func(pop[i])
            calls_made += 1

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]

        # Main loop
        while calls_made < budget:
            for i in range(NP):
                if calls_made >= budget:
                    break
                
                # Select three distinct random individuals
                # Ensure r1, r2, r3 are distinct and not i? 
                # Standard DE/rand/1 allows r1,r2,r3 to be any, but usually distinct from each other.
                # Often r1, r2, r3 are chosen from population, can include i? 
                # Usually "distinct from each other".
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Boundary handling
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
