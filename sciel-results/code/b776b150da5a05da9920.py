import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 300  # Safety margin for 330s limit

    def __call__(self, func):
        rng = np.random.default_rng(42)
        N = 20  # Population size
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        # Initialize population using Sobol for better coverage
        # scipy.stats.qmc.Sobol
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        points = sampler.random(N)
        # Map from [0, 1] to [-5, 5]
        points = points * 10.0 - 5.0
        
        pop = points
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        evals_used = N
        remaining = self.budget - evals_used
        
        # Differential Evolution loop
        # We will evaluate one individual at a time to respect the budget strictly
        # and allow for early stopping if time runs out.
        
        while evals_used < self.budget:
            if time.time() > self.deadline:
                break
                
            # Select a random index for the new candidate
            # We will replace the worst individual or a random one?
            # Standard DE replaces the target if the trial is better.
            # To keep it simple and sequential:
            # Pick a target index i.
            # Generate trial vector.
            # Evaluate trial.
            # If trial < pop[i], replace.
            
            i = rng.integers(0, N)
            # Choose 3 distinct indices different from i
            while True:
                a, b, c = rng.integers(0, N, size=3)
                if a != b and b != c and c != a and a != i and b != i and c != i:
                    break
            
            # DE/rand/1 mutation
            v = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            r = rng.random(self.dim)
            j_rand = rng.integers(0, self.dim)
            u = np.where(r <= CR, v, pop[i])
            u[j_rand] = v[j_rand]
            
            # Clip to bounds
            u = np.clip(u, -5.0, 5.0)
            
            # Evaluate
            fu = func(u)
            evals_used += 1
            
            if fu < fitness[i]:
                pop[i] = u
                fitness[i] = fu
                if fu < best_f:
                    best_f = fu
                    best_x = u.copy()
                    
        return best_x, best_f
