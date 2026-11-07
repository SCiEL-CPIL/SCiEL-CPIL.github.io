"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution Optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # DE Parameters
        # Population size N=20 allows 25 generations in 500 calls.
        # This balances global exploration (diversity) with sufficient iterations.
        self.N = 20
        self.F = 0.8  # Mutation factor
        self.CR = 0.5 # Crossover rate

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Adjust population size if budget is small
        N = min(self.N, budget)
        
        if N < 3:
            # Fallback for very small budgets: Random Search
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            for _ in range(budget - 1):
                y = rng.uniform(-5, 5, dim)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
            return x, fx

        # Initialize Population
        X = rng.uniform(-5, 5, (N, dim))
        F_vals = np.array([func(x) for x in X])
        
        best_idx = np.argmin(F_vals)
        best_x = X[best_idx].copy()
        best_f = F_vals[best_idx]
        
        calls_made = N
        start_time = time.time()
        # Safety margin: 330s limit, stop at 280s
        deadline = start_time + 280.0
        
        while calls_made + N <= budget:
            if time.time() > deadline:
                break
                
            # Pre-generate indices for this generation to reduce overhead
            # We need 3 distinct indices different from i for each i
            # Doing this inside the loop is simpler and fast enough for N=20
            new_X = X.copy()
            new_F_vals = F_vals.copy()
            
            for i in range(N):
                # Select 3 distinct random indices excluding i
                # Using choice with replace=False
                candidates = [j for j in range(N) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + self.F * (X[r2] - X[r3])
                
                # Binomial Crossover
                # j_rand ensures at least one component comes from v
                j_rand = rng.integers(dim)
                u = np.empty(dim)
                
                # Vectorized crossover mask
                mask = rng.random(dim) < self.CR
                mask[j_rand] = True # Ensure at least one
                
                u[mask] = v[mask]
                u[~mask] = X[i][~mask]
                
                # Clip to box [-5, 5]
                np.clip(u, -5, 5, out=u)
                
                # Evaluate
                f_u = func(u)
                calls_made += 1
                
                # Selection (1-to-1 tournament)
                if f_u <= new_F_vals[i]:
                    new_X[i] = u
                    new_F_vals[i] = f_u
                    
                if f_u < best_f:
                    best_f = f_u
                    best_x = u.copy()
                    
            X = new_X
            F_vals = new_F_vals

        return best_x, best_f
# EVOLVE-BLOCK-END
