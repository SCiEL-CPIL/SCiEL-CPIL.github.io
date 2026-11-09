import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        N = 20
        if self.budget < N:
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx
            
        # Initialize swarm
        X = rng.uniform(-5, 5, (N, self.dim))
        V = rng.uniform(-1, 1, (N, self.dim))
        
        fX = np.array([func(x) for x in X])
        
        pbest_X = X.copy()
        pbest_f = fX.copy()
        gbest_idx = np.argmin(fX)
        gbest_X = X[gbest_idx].copy()
        gbest_f = fX[gbest_idx]
        
        w_start = 0.9
        w_end = 0.4
        c1 = 1.49445
        c2 = 1.49445
        
        calls_made = N
        
        total_main_budget = self.budget - N
        if total_main_budget > 0:
             max_gens = total_main_budget // N
             
             for g in range(max_gens):
                 # Adaptive inertia
                 if max_gens > 1:
                     progress = g / (max_gens - 1)
                 else:
                     progress = 0.0
                 w = w_start - (w_start - w_end) * progress
                 
                 for i in range(N):
                     r1 = rng.random(self.dim)
                     r2 = rng.random(self.dim)
                     
                     # Update velocity
                     V[i] = w * V[i] + c1 * r1 * (pbest_X[i] - X[i]) + c2 * r2 * (gbest_X - X[i])
                     
                     # Update position
                     X[i] = X[i] + V[i]
                     
                     # Clip to bounds
                     X[i] = np.clip(X[i], -5, 5)
                     
                     # Evaluate
                     fX[i] = func(X[i])
                     calls_made += 1
                     
                     # Update personal best
                     if fX[i] < pbest_f[i]:
                         pbest_f[i] = fX[i]
                         pbest_X[i] = X[i].copy()
                         
                         # Update global best
                         if fX[i] < gbest_f:
                             gbest_f = fX[i]
                             gbest_X = X[i].copy()
                             
        # Use remaining calls for local search around gbest
        remaining = self.budget - calls_made
        if remaining > 0:
             for _ in range(remaining):
                 dx = rng.normal(0, 0.1, self.dim)
                 x_new = np.clip(gbest_X + dx, -5, 5)
                 f_new = func(x_new)
                 if f_new < gbest_f:
                     gbest_f = f_new
                     gbest_X = x_new.copy()
                     
        return gbest_X, gbest_f
