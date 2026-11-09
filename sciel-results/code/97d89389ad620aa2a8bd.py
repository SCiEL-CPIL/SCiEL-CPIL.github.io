"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        time_limit = 300
        
        rng = np.random.default_rng(0)
        
        # DE Parameters
        NP = 10  # Population size
        F = 0.8   # Differential weight
        CR = 0.9  # Crossover rate
        
        # Calculate max generations to fit budget
        # Initial evals: NP
        # Each gen: NP evals
        # Total = NP + G * NP <= budget
        # G <= (budget - NP) / NP
        max_gens = (budget - NP) // NP
        
        # Initialize population
        population = rng.uniform(-5, 5, (NP, dim))
        fvals = np.array([func(x) for x in population])
        
        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Main Loop
        for g in range(max_gens):
            if time.time() - start_time > time_limit:
                break
                
            # Create new population
            new_population = np.zeros((NP, dim))
            new_fvals = np.zeros(NP)
            
            for i in range(NP):
                # Select 3 distinct random indices
                # Use rng.choice for efficiency
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                # Ensure they are distinct from i? 
                # Standard DE/rand/1/bin usually allows r1,r2,r3 to be any, 
                # but often excludes i to avoid self-mutation.
                # If r1==i, v = x[i] + F(x[r2]-x[r3]), which is valid.
                # But let's ensure r1,r2,r3 are distinct from each other.
                # rng.choice with replace=False ensures distinct.
                
                # Mutation
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Crossover
                trial = population[i].copy()
                j_rand = rng.integers(dim)
                # Vectorized crossover
                mask = rng.random(dim) < CR
                mask[j_rand] = True
                trial[mask] = v[mask]
                
                # Clip
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                ft = func(trial)
                
                # Selection
                if ft <= fvals[i]:
                    new_population[i] = trial
                    new_fvals[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                else:
                    new_population[i] = population[i]
                    new_fvals[i] = fvals[i]
            
            population = new_population
            fvals = new_fvals
            
        return best_x, best_f
# EVOLVE-BLOCK-END
