"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        
        # Population size
        pop_size = 10
        F = 0.8  # Mutation factor
        CR = 0.5 # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        evals = np.empty(pop_size)
        for i in range(pop_size):
            evals[i] = func(pop[i])
        
        budget_used = pop_size
        best_idx = np.argmin(evals)
        best_x = pop[best_idx].copy()
        best_f = evals[best_idx]
        
        # DE loop
        while budget_used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate trial vector
            # Need 3 distinct indices different from i
            i = rng.integers(pop_size)
            others = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(others, size=3, replace=False)
            
            # Mutation: DE/rand/1
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Clip to bounds
            v = np.clip(v, -5, 5)
            
            # Crossover: Binomial
            r = rng.random()
            j_rand = rng.integers(dim)
            trial = np.where(rng.random(dim) < CR, v, pop[i])
            trial[j_rand] = v[j_rand]  # Ensure at least one component from v
            
            # Evaluate trial
            f_trial = func(trial)
            budget_used += 1
            
            # Selection
            if f_trial < evals[i]:
                pop[i] = trial
                evals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
