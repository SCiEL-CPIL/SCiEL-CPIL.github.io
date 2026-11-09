"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Stochastic Differential Evolution with restarts for BBOB f4 (Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, 10s safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Global state
        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(123)

        def wrapped_func(x):
            nonlocal call_count, global_best_x, global_best_f
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x_clipped.copy()
            return f_val

        # Differential Evolution Parameters
        pop_size = 20
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        # Spread initial population across the box to ensure coverage
        population = np.empty((pop_size, dim))
        for i in range(pop_size):
            population[i] = rng.uniform(-5, 5, dim)
        
        # Evaluate initial population
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            fitness[i] = wrapped_func(population[i])
            
        # Identify initial best
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_f = fitness[best_idx]

        # Main DE Loop
        while call_count < budget and time.time() < deadline:
            # Check for stagnation: if no improvement in last 50 evaluations, restart
            # We track this simply by checking if best_f hasn't changed significantly 
            # and we've done a chunk of evaluations.
            # For simplicity in this structural change, we rely on the diversity 
            # of DE to escape. If we want explicit restarts:
            # (Omitted complex restart logic for now to keep code robust and fast, 
            # DE with F=0.8, CR=0.9 is generally robust enough for 5D Rastrigin 
            # to find the global min if initialized well, but let's add a 
            # simple "reset" if no improvement in 100 evals)
            
            # Actually, let's implement a simple restart mechanism:
            # If we haven't improved global_best_f in the last 50 calls, 
            # we might be stuck. 
            # To keep it simple and within the "structural change" directive, 
            # I will implement a basic DE loop. 
            # If stagnation is detected (best_f unchanged for 50 steps), 
            # we re-initialize the population around the current best with larger spread.
            
            no_improve_count = 0
            last_best_f = global_best_f
            
            # Generate one offspring per generation (or batch?)
            # Budget is 500. Pop size 20. 
            # Standard DE evaluates pop_size offspring per generation.
            # 500 / 20 = 25 generations. This is quite few for DE.
            # Maybe use a smaller pop size or (1+lambda) style?
            # Let's stick to a (mu+lambda) like behavior or just standard DE 
            # but with a smaller population to allow more generations?
            # Or just run DE until budget runs out.
            
            # Let's use a population of 10 to get ~50 generations.
            # Re-initialize if we haven't done so.
            # Wait, I already initialized pop_size=20. 
            # Let's adjust: Use pop_size=10 for more generations.
            # But I already evaluated 20 points. 
            # Let's just proceed with the current population. 
            # 500 calls / 10 pop = 50 gens.
            # 500 calls / 20 pop = 25 gens.
            # 25 generations of DE in 5D might be enough if F is high.
            
            # Let's switch to a smaller population for this run to allow more exploration steps.
            # I will re-initialize with pop_size=10 if I haven't started the loop yet?
            # No, I already spent 20 calls. 
            # Let's just use the existing population of 20. 
            # 25 generations is acceptable.
            
            while call_count < budget and time.time() < deadline:
                # Select base vector (best/1 strategy: use current global best)
                base_idx = np.argmin(fitness)
                base = population[base_idx]
                base_f = fitness[base_idx]
                
                # Pick two distinct random vectors
                r1, r2 = np.random.choice(pop_size, 2, replace=False)
                while r1 == base_idx or r2 == base_idx:
                    r1, r2 = np.random.choice(pop_size, 2, replace=False)
                    
                v = population[base_idx] + F * (population[r1] - population[r2])
                
                # Binary Crossover
                trial = np.empty(dim)
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[base_idx][j]
                        
                trial = np.clip(trial, -5, 5)
                trial_f = wrapped_func(trial)
                
                # Selection
                if trial_f <= fitness[base_idx]:
                    population[base_idx] = trial
                    fitness[base_idx] = trial_f
                    
                # Check stagnation
                if global_best_f < last_best_f - 1e-10:
                    last_best_f = global_best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # If stuck for too long (e.g., 50 evaluations), restart population
                if no_improve_count > 50:
                    # Re-initialize population around current best
                    for i in range(pop_size):
                        population[i] = global_best_x + rng.normal(0, 1.0, dim)
                        population[i] = np.clip(population[i], -5, 5)
                        fitness[i] = wrapped_func(population[i])
                    no_improve_count = 0
                    last_best_f = global_best_f

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
