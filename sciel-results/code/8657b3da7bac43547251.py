"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid optimizer: Differential Evolution for global search, (1+1)-ES for local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            calls_made += 1
            return func(np.clip(x, -5, 5))

        # --- Phase 1: Differential Evolution (Global Search) ---
        # Population size and parameters
        NP = 30  # Population size
        F = 0.8  # Scale factor
        CR = 0.5 # Crossover rate
        
        # Reserve some budget for local refinement
        # Each generation costs NP calls.
        # Let's use 60% of budget for DE.
        de_budget = int(budget * 0.6)
        max_generations = de_budget // NP
        
        # Initialize population
        # Mix of random points and points near the center to ensure coverage
        population = []
        for i in range(NP):
            if i == 0:
                x = np.zeros(n)
            elif i < NP // 2:
                x = rng.uniform(-2, 2, n)
            else:
                x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            population.append((x, fx))
            
        # Sort initial population
        population.sort(key=lambda p: p[1])
        best_x, best_fx = population[0]

        # DE Loop
        for gen in range(max_generations):
            if calls_made >= budget or time.time() > deadline:
                break
                
            new_population = []
            for i in range(NP):
                # Select distinct random indices
                r1, r2, r3 = rng.choice(NP, size=3, replace=False)
                
                # Mutation: DE/rand/1
                v = population[r1][0] + F * (population[r2][0] - population[r3][0])
                
                # Crossover: Binomial
                trial = np.copy(population[i][0])
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = evaluate(trial)
                
                # Selection: Greedy
                if f_trial < population[i][1]:
                    new_population.append((trial, f_trial))
                else:
                    new_population.append(population[i])
                
                # Update global best
                if f_trial < best_fx:
                    best_fx = f_trial
                    best_x = trial.copy()

            # Replace population
            population = new_population
            population.sort(key=lambda p: p[1])

        # --- Phase 2: Local Refinement (Exploitation) ---
        # Use remaining budget for a fine-grained (1+1)-ES starting from the DE best
        if calls_made < budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.5  # Start with a smaller step size for refinement
            no_improve = 0
            max_no_improve = 20
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                    
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    sigma *= 1.1  # Expand slightly on success
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma *= 0.9  # Shrink on failure
                    if no_improve >= max_no_improve:
                        # If stuck, try a small random jump
                        x = np.clip(x + 0.1 * rng.standard_normal(n), -5, 5)
                        fx = evaluate(x)
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        no_improve = 0
                        sigma = 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
