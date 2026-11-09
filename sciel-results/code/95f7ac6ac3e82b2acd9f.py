import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We aim to finish well before.
        # 500 calls is fast, but let's be safe.
        deadline = start_time + 300
        
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            
            # Clip to box [-5, 5]
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Differential Evolution Parameters
        # Population size: small for 5D, but enough for diversity. 
        # 500 calls total. 
        # If NP=10, 50 generations. 
        # If NP=20, 25 generations.
        # Let's try NP=15. 500/15 ~ 33 generations.
        NP = 15
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        # Start with a mix: one at 0, others random in [-5, 5]
        # Actually, for Rosenbrock, starting near 0 is often good if the shift isn't huge.
        # But the shift is in [-4, 4]. So 0 is likely close.
        # Let's initialize uniformly in [-5, 5] to ensure coverage, 
        # but maybe bias towards center?
        # The task says minimizer lies in [-4, 4]^5.
        # Uniform [-5, 5] covers it.
        
        population = rng.uniform(-5, 5, (NP, n))
        # Force one individual to be zero? Or just let DE handle it.
        # Let's keep it random for diversity.
        
        fitness = np.array([evaluate(ind) for ind in population])
        
        # DE Loop
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Check if we have enough budget for a full generation
            # A generation costs NP evaluations.
            if calls_made + NP > budget:
                # Do partial generation or break
                # We can just break if we can't do a full generation, 
                # or do a few random evaluations to use up budget?
                # Better to just break and let the loop end.
                # However, "repeating the last n_i if fewer calls are made" means
                # wasting the last few calls on random noise is bad if it doesn't improve.
                # But if we stop early, we save time.
                # Let's just break if we can't complete a generation.
                break

            for i in range(NP):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    # Re-sample if any match i
                    r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                # Create trial vector
                v = population[r1] + F * (population[r2] - population[r3])
                
                # Clip v to box? 
                # DE usually works on unbounded, but our function clips.
                # It's better to let the function clip, but for the crossover,
                # we should probably clip v to avoid weird jumps?
                # Actually, the crossover creates u. We evaluate u.
                
                # Crossover
                jrand = rng.integers(n)
                u = np.copy(population[i])
                for j in range(n):
                    if rng.random() < CR or j == jrand:
                        u[j] = v[j]
                
                # Evaluate trial
                f_u = evaluate(u)
                
                # Selection
                if f_u <= fitness[i]:
                    population[i] = u
                    fitness[i] = f_u

        return best_x, best_f
