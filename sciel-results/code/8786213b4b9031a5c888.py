"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # DE Parameters
        NP = 20  # Population size
        F = 0.8  # Mutation scale factor
        CR = 0.9 # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, size=(NP, n))
        
        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        calls = NP
        
        # Main loop
        while calls < budget:
            # Check if we can afford a full generation
            if NP > budget - calls:
                # If not enough budget for full population, do a few random restarts or local steps?
                # For simplicity, break or do a smaller step. 
                # But DE requires a full population for mutation.
                # Let's just break if we can't do a full generation.
                break
                
            new_pop = np.zeros_like(pop)
            new_fvals = np.zeros(NP)
            
            for i in range(NP):
                # Select 3 distinct random indices
                r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                while r1 == r2 or r2 == r3 or r1 == r3:
                    r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                        
                # Bound handling: Reflect or Clip
                # Clip is simple
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    new_pop[i] = trial
                    new_fvals[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fvals[i] = fvals[i]
                    
            pop = new_pop
            fvals = new_fvals
            
            # Update best from population (just in case, though we tracked it)
            # Actually, we updated best_x in the loop.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
