"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution (DE) implementation for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size N. For dim=5, N=10-20 is standard.
        # Budget is 500. 
        # Initial population: N calls.
        # Remaining budget: 500 - N.
        # Each generation: N calls.
        # Let N = 10. Generations = (500 - 10) / 10 = 49.
        # Total calls = 10 + 49 * 10 = 500.
        N = 10
        
        # Differential weight F and crossover rate CR
        F = 0.8
        CR = 0.9
        
        rng = np.random.default_rng(42)
        
        # Initialize population uniformly in [-5, 5]
        # This ensures global coverage immediately, which helps AOCC early on
        # if the minimum is far from 0.
        pop = rng.uniform(-5, 5, size=(N, dim))
        fitness = np.array([func(x) for x in pop])
        
        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        calls_made = N
        
        # Main loop
        while calls_made < budget:
            if time.time() > deadline - 5.0:
                break
                
            # Check if we can afford a full generation
            if calls_made + N > budget:
                # If we can't do a full generation, do random sampling with remaining budget?
                # Or just break? 
                # Since N=10 and budget=500, this won't happen unless budget isn't divisible.
                # But to be safe:
                remaining = budget - calls_made
                if remaining <= 0:
                    break
                # Do 'remaining' random steps around best? 
                # Actually, simpler to just break if we can't do a full DE step, 
                # but we can use the remaining calls for a local (1+1) polish?
                # Let's just break to keep logic simple and safe.
                break

            # Generate offspring for each individual in population
            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(N)
            
            for i in range(N):
                # Select 3 distinct random indices different from i
                # Use a simple loop to ensure distinctness
                while True:
                    a, b, c = np.random.choice(N, 3, replace=False)
                    if a != i and b != i and c != i:
                        break
                
                # Differential mutation
                # v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Clip to box [-5, 5]
                v = np.clip(v, -5, 5)
                
                # Binomial Crossover
                # Trial vector u
                u = pop[i].copy()
                # Random position to start crossover
                j_rand = rng.integers(0, dim)
                
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Ensure u is within bounds (it is, because v is clipped and pop[i] is within bounds)
                
                # Evaluate
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u <= fitness[i]:
                    new_pop[i] = u
                    new_fitness[i] = f_u
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Update population
            pop = new_pop
            fitness = new_fitness
            
            # Update global best
            curr_best_idx = np.argmin(fitness)
            curr_best_f = fitness[curr_best_idx]
            
            if curr_best_f < best_f:
                best_f = curr_best_f
                best_x = pop[curr_best_idx].copy()
            
            # Optional: Adaptive F and CR? 
            # For simplicity and robustness, keep fixed. 
            # However, if stagnation is detected, we could restart or increase F.
            # Given the strict budget and "different strategy" directive, 
            # a robust standard DE is the core change.
            
            # Stagnation check: If no improvement in last 5 generations, 
            # we might be stuck. 
            # A simple strategy: if stuck, re-seed the worst half of the population randomly.
            # This acts as a form of multi-start within DE.
            # Let's track improvement.
            # We don't have a history array here easily, so let's add a simple counter.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
