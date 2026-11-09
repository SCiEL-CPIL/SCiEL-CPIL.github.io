"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive scaling and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 290s guard for 300s limit
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 20
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        calls_made = pop_size
        
        # Track progress for AOCC (we don't need to return this, but we track best_f)
        # The evaluator tracks b_i (best value after call i). 
        # Since we call func in batches, the evaluator sees them sequentially.
        # We just need to minimize best_f over the 500 calls.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Check if we should stop early if we are very close to 0? 
            # No, keep going to use full budget if possible, unless time runs out.
            
            # Adaptive F: increase if no improvement recently? 
            # Simple DE usually keeps F constant. Let's stick to standard DE with slight adaptation.
            # For robustness, we'll use a fixed F=0.8, CR=0.9 which is standard.
            
            # Generate offspring
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                    
                # Select two distinct random indices different from i
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i:
                    r1 = rng.choice(pop_size)
                while r2 == i or r2 == r1:
                    r2 = rng.choice(pop_size)
                
                # Mutation: v = pop[r1] + F * (pop[r2] - pop[i])
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Bound handling: clip
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                ft = func(trial)
                calls_made += 1
                
                # Selection
                if ft < fvals[i]:
                    pop[i] = trial
                    fvals[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        
        return best_x, best_f
# EVOLVE-BLOCK-END
