import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin from 330s limit
        
        # DE Parameters
        # Budget is 300. 
        # Initial population: 10
        # Each generation: 10 evaluations
        # 300 / 10 = 30 generations.
        # This fits perfectly into the budget without exceeding it.
        N = 10  # Population size
        F = 0.8  # Differential weight
        CR = 0.5  # Crossover rate
        
        dim = self.dim
        
        # Initialize population uniformly in [-5, 5]^dim
        # Note: The problem states minimizer is in [-4, 4]^5, but search space is [-5, 5]^5.
        # Uniform init covers the space.
        pop = rng.uniform(-5, 5, (N, dim))
        
        # Evaluate initial population
        fpop = np.array([func(x) for x in pop])
        
        # Track best
        best_idx = np.argmin(fpop)
        x_best = pop[best_idx].copy()
        f_best = fpop[best_idx]
        
        calls_made = N
        
        # Main Loop
        # We have budget 300. We used N=10. Remaining: 290.
        # We can run 29 more generations (29 * 10 = 290).
        # Total calls: 10 + 290 = 300.
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if self.budget - calls_made < N:
                break
                
            for i in range(N):
                # Select three distinct random indices different from i
                # Efficient way: pick 3 random indices, ensure they are not i and distinct from each other
                # For N=10, this is fast enough.
                r1, r2, r3 = i, i, i
                while r1 == i:
                    r1 = rng.integers(N)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(N)
                while r3 == i or r3 == r1 or r3 == r2:
                    r3 = rng.integers(N)
                
                # Create trial vector
                # DE/rand/1
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                j_rand = rng.integers(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= fpop[i]:
                    pop[i] = trial
                    fpop[i] = f_trial
                    
                    if f_trial < f_best:
                        f_best = f_trial
                        x_best = trial.copy()
            
            # Optional: Check for stagnation and restart if needed?
            # DE usually handles this well with population diversity.
            # However, if f_best hasn't improved in many generations, we might want to inject randomness.
            # Given the small budget (300), standard DE is likely sufficient.
            # If we want to be safe against local optima in multimodal functions (Rastrigin/Gallagher),
            # we can monitor improvement.
            
            # Let's add a simple stagnation check: if no improvement in last 5 generations,
            # replace the worst 2 individuals with random ones.
            # This requires tracking history. 
            # To keep it simple and robust, we'll just let DE run. 
            # DE's diversity usually prevents total stagnation in 5D.
            
            # If we want to ensure we don't waste time on a stuck population:
            # We can check if f_best changed in the last generation.
            # But tracking this adds complexity. Let's stick to standard DE first.
            # If score is low, we might need restarts.
            # For now, standard DE.

        return x_best, f_best
