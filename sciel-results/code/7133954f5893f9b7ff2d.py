import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Differential Evolution strategy
        # Pop size N=10, CR=0.5, F=0.8
        N = 10
        F = 0.8
        CR = 0.5
        
        rng = np.random.default_rng(12345)
        
        start_time = time.time()
        deadline = start_time + 320.0
        
        # Initialize population uniformly in [-5, 5]^dim
        # This uses N evaluations
        pop = rng.uniform(-5, 5, size=(N, self.dim))
        f_pop = np.array([func(x) for x in pop])
        
        calls_made = N
        
        # Track best
        best_idx = np.argmin(f_pop)
        x_best = pop[best_idx].copy()
        f_best = f_pop[best_idx]
        
        # If budget is too small, just return best
        if calls_made >= self.budget:
            return x_best, f_best
            
        # Main loop
        while calls_made < self.budget:
            # Time check
            if time.time() > deadline:
                break
                
            # Generate offspring for each individual in population
            # We can only evaluate one at a time to strictly respect budget and time
            # But we can generate all candidates first and then evaluate?
            # No, we must check budget.
            # We will iterate through the population and generate one candidate per individual per generation.
            # If budget runs out mid-generation, we stop.
            
            # To be efficient and safe, we process one candidate at a time.
            # However, DE usually works in generations.
            # Let's do a "semi-online" DE:
            # For each i in 0..N-1:
            #   Generate trial vector v_i
            #   Evaluate f(v_i)
            #   If f(v_i) < f_pop[i], replace pop[i]
            
            # We need to track which indices have been processed in the current generation
            # to avoid using updated individuals for generating others in the same generation (standard DE)
            # vs using them (online DE). Standard DE is usually better for stability.
            # So we keep a copy of the population for the source vectors.
            
            pop_old = pop.copy()
            f_pop_old = f_pop.copy()
            
            for i in range(N):
                if calls_made >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Select three distinct random indices different from i
                # Simple way: sample 3 indices, ensure they are unique and != i
                # Since N=10, we can just sample and retry if collision, or use permutation
                indices = [i]
                while len(indices) < 4:
                    j = rng.integers(0, N)
                    if j not in indices:
                        indices.append(j)
                        
                a, b, c = indices[1], indices[2], indices[3]
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop_old[a] + F * (pop_old[b] - pop_old[c])
                
                # Crossover
                r = rng.random(self.dim)
                j_rand = rng.integers(0, self.dim)
                u = np.where(r < CR, v, pop_old[i])
                u[j_rand] = v[j_rand]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluation
                f_u = func(u)
                calls_made += 1
                
                # Selection
                if f_u < f_pop_old[i]:
                    pop[i] = u
                    f_pop[i] = f_u
                    
                    if f_u < f_best:
                        f_best = f_u
                        x_best = u.copy()
            
            # After a full generation, update the "old" references if we completed it?
            # Actually, we just loop. The next iteration will take a new snapshot.
            # If we broke early due to budget, the loop condition handles exit.
            
            # Optional: If no improvement for many generations, we could restart or perturb
            # But DE with N=10 is fairly robust. 
            # Let's add a simple stagnation check: if no global best improvement in 50 calls, 
            # maybe inject a random point? 
            # The budget is only 300. 50 calls is significant.
            # Let's just rely on DE's exploration.
            
        return x_best, f_best
