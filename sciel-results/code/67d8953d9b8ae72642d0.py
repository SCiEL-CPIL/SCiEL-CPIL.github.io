"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with global restarts for robustness against multimodality."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE Parameters
        pop_size = 10
        F = 0.8
        CR = 0.5

        # Global state
        global_best_x = np.zeros(dim)
        global_best_f = float('inf')

        # Initial evaluation
        x0 = np.zeros(dim)
        fx0 = func(x0)
        if fx0 < global_best_f:
            global_best_f = fx0
            global_best_x = x0.copy()

        calls_made = 1

        # Main loop: DE with restarts
        # We divide the budget into segments. If no improvement in a segment, restart.
        # DE is stochastic and handles non-separability better than simple ES.
        # Schwefel has many local minima, so restarts are crucial.
        
        # Strategy: Run DE for a fixed number of evaluations, check for improvement.
        # If stuck, restart with a new population around the current best (or global random).
        
        # Let's use a "sliding window" of evaluations.
        # Actually, standard DE maintains a population. 
        # Let's allocate the budget to DE iterations.
        # With dim=5 and pop_size=10, each iteration costs 10 evaluations.
        # 500 budget / 10 = 50 generations.
        
        # To handle multimodality, we will restart if the best value doesn't improve
        # for a certain number of generations, or if we hit a local minimum.
        # Given the small budget, simple DE might get stuck.
        # Let's implement a simple restart policy:
        # Every 100 evaluations, if no significant improvement, restart population.
        
        # A more robust approach for small budgets:
        # Use a mixture of local search (DE) and global exploration (random restarts).
        
        # Let's try a specific strategy:
        # 1. Initialize population.
        # 2. Run DE for N generations.
        # 3. If best_f hasn't improved by a factor (or amount) in last N gens, 
        #    re-seed population around global_best_x with some noise, or randomly.
        
        # Given 500 evals, let's do 5 restarts of 100 evals each?
        # Or just run DE for 500 evals but with aggressive mutation when stuck?
        
        # Let's stick to a clean DE implementation with a "stagnation" check.
        
        # Initialize Population
        # Uniform in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        
        # Evaluate initial population
        pop_f = np.array([func(x) for x in pop])
        calls_made += pop_size
        
        # Update global best
        min_idx = np.argmin(pop_f)
        if pop_f[min_idx] < global_best_f:
            global_best_f = pop_f[min_idx]
            global_best_x = pop[min_idx].copy()

        # Sort population by fitness (best first)
        # We'll maintain the population sorted for simplicity, though DE doesn't strictly require it
        # but it helps for tracking the best.
        
        # DE Loop
        gen = 0
        no_improve_count = 0
        best_f_this_run = global_best_f
        
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Check stagnation
            if no_improve_count > 10: # 10 generations without improvement
                # Restart strategy: 
                # 50% chance to keep current best and add noise
                # 50% chance to fully random restart
                if rng.random() < 0.5:
                    # Local restart: perturb global best
                    noise = rng.normal(0, 1.0, (pop_size, dim))
                    pop = global_best_x + noise
                    pop = np.clip(pop, -5, 5)
                else:
                    # Global restart
                    pop = rng.uniform(-5, 5, (pop_size, dim))
                
                # Re-evaluate
                pop_f = np.array([func(x) for x in pop])
                calls_made += pop_size
                
                min_idx = np.argmin(pop_f)
                if pop_f[min_idx] < global_best_f:
                    global_best_f = pop_f[min_idx]
                    global_best_x = pop[min_idx].copy()
                    no_improve_count = 0
                else:
                    # Still no improvement, but we've refreshed the population
                    # Don't increment no_improve_count here immediately, give it a few gens
                    pass 
                    
                continue # Skip to next generation loop logic
            
            # Standard DE Step
            new_pop = np.zeros_like(pop)
            new_f = np.zeros(pop_size)
            
            # We process one individual at a time to check budget
            for i in range(pop_size):
                if calls_made >= budget:
                    break
                    
                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= pop_f[i]:
                    new_pop[i] = trial
                    new_f[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_f[i] = pop_f[i]
            
            # Update population
            pop = new_pop
            pop_f = new_f
            
            # Check for improvement
            min_idx = np.argmin(pop_f)
            if pop_f[min_idx] < global_best_f:
                global_best_f = pop_f[min_idx]
                global_best_x = pop[min_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            gen += 1

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
