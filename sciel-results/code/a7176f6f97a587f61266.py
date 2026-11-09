import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 330s limit, keep 30s margin
        start_time = time.time()
        deadline = start_time + 300.0

        rng = np.random.default_rng(42)
        
        # Population size
        # 500 budget. N=10 allows 50 generations.
        # N=20 allows 25 generations.
        # For 5D, N=10 is a good balance.
        N = 10
        if self.budget < N:
            # Fallback to simple random search if budget is tiny
            best_x = None
            best_f = np.inf
            for _ in range(self.budget):
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x
            return best_x, best_f

        # Parameters
        F = 0.5  # Differential weight
        CR = 0.5 # Crossover rate

        # Initialize population
        pop = rng.uniform(-5, 5, (N, self.dim))
        pop_fx = np.array([func(x) for x in pop])
        
        # Track global best
        best_idx = np.argmin(pop_fx)
        best_x = pop[best_idx].copy()
        best_f = pop_fx[best_idx]
        
        calls_made = N
        
        # Main loop
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Determine how many full generations we can do
            # We need N calls per generation
            remaining = self.budget - calls_made
            if remaining < N:
                # Do partial generation or just break
                # It's better to break to keep the population consistent, 
                # or we can just evaluate the remaining ones against the current best
                # For simplicity and robustness, if we can't do a full generation, 
                # we just break. The last full generation's best is our result.
                break
            
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices
                # Exclude i
                candidates = np.setdiff1d(np.arange(N), i)
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Create trial vector
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Binomial Crossover
                # Randomly decide which components come from v and which from pop[i]
                # Ensure at least one component from v
                j0 = rng.integers(0, self.dim)
                u = np.empty(self.dim)
                r = rng.random(self.dim)
                
                # Standard DE crossover
                # u[j] = v[j] if r[j] < CR or j == j0 else pop[i][j]
                mask = (r < CR) | (np.arange(self.dim) == j0)
                u[mask] = v[mask]
                u[~mask] = pop[i][~mask]
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                fu = func(u)
                calls_made += 1
                
                # Selection
                if fu < pop_fx[i]:
                    pop[i] = u
                    pop_fx[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()

        return best_x, best_f
