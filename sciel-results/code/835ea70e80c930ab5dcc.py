"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution Optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size: typically 5*dim to 10*dim. 5*5=25 is a good balance.
        pop_size = max(10, 5 * dim) 
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        # Use a fixed seed for reproducibility if needed, but random is fine for DE
        rng = np.random.default_rng(0)
        pop = rng.uniform(-5, 5, (pop_size, dim))
        
        # Evaluate initial population
        fvals = np.array([func(x) for x in pop])
        calls_made = pop_size
        
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Adaptive F and CR? 
        # For simplicity and robustness, we stick to standard parameters but 
        # could add a simple adaptation if stagnation is detected.
        # Given the budget is small (500 calls), fixed params are safer to avoid 
        # wasting time on adaptation logic.
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline - 1.0:
                break
                
            # Generate new population
            new_pop = np.zeros((pop_size, dim))
            new_fvals = np.zeros(pop_size)
            
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                if time.time() > deadline - 1.0:
                    break
                
                # Select two distinct random indices j, k different from i
                # and different from best_idx (to avoid trivial mutation)
                # Simple approach: pick random j, k. If they collide, pick again.
                # For efficiency, we can just pick random indices and handle collisions loosely 
                # or just pick from a pre-shuffled list. 
                # Given small pop_size, simple random picking is fast enough.
                
                j, k = i, i
                while j == i or k == i or j == k:
                    j = rng.integers(0, pop_size)
                    k = rng.integers(0, pop_size)
                
                # Mutation: v = pop[best_idx] + F * (pop[j] - pop[k])
                # This is "Current-to-Best/1" variant if we used pop[i] instead of pop[best], 
                # but "Best/1" is often stronger for convergence. 
                # Let's use "Best/1" for aggressive exploitation, 
                # or "Rand/1" for exploration. 
                # A mix is good. Let's use "Best/1" to drive towards the best found.
                v = pop[best_idx] + F * (pop[j] - pop[k])
                
                # Clip to bounds
                v = np.clip(v, -5, 5)
                
                # Crossover
                trial = pop[i].copy()
                r = rng.random()
                j_rand = rng.integers(0, dim)
                
                for d in range(dim):
                    if rng.random() <= CR or d == j_rand:
                        trial[d] = v[d]
                    else:
                        trial[d] = pop[i][d]
                
                # Evaluation
                ft = func(trial)
                calls_made += 1
                
                new_pop[i] = trial
                new_fvals[i] = ft
                
                # Update best
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
                    best_idx = i # Note: best_idx in pop array might change if we replace pop
                    
            # If we broke early due to budget/time, we might not have a full new_pop
            # But DE requires a full generation for proper selection. 
            # If we ran out of budget mid-generation, we just stop.
            
            # Selection
            # Only replace if the trial is better
            for i in range(pop_size):
                if new_fvals[i] < fvals[i]:
                    pop[i] = new_pop[i]
                    fvals[i] = new_fvals[i]
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()
                        best_idx = i
            
            # Optional: If no improvement in a while, we could restart or perturb.
            # But DE with CR=0.9 and F=0.8 is usually robust enough.
            # Given the 300s limit and 500 calls, we should just let it run.

        return best_x, best_f
# EVOLVE-BLOCK-END
