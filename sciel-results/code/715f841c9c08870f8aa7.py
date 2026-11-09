import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        lb = -5.0
        ub = 5.0
        
        # Choose number of particles
        # Let's use 10 particles
        n_particles = 10
        # Total calls available for iterations
        # Init: n_particles
        # Iter: n_particles * n_iters
        # n_particles + n_particles * n_iters <= budget
        # n_iters = (budget - n_particles) // n_particles
        
        n_iters = (budget - n_particles) // n_particles
        
        rng = np.random.default_rng(42)
        
        # Initialize positions
        X = rng.uniform(lb, ub, (n_particles, n))
        V = rng.uniform(-1.0, 1.0, (n_particles, n)) # Initial velocities
        
        # Evaluate initial positions
        F = np.array([func(x) for x in X])
        calls = n_particles
        
        # Initialize personal bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # Initialize global best
        gbest_idx = np.argmin(Pbest_F)
        Gbest_X = Pbest_X[gbest_idx].copy()
        Gbest_F = Pbest_F[gbest_idx]
        
        w = 0.9 # Inertia
        w_min = 0.4
        c1 = 1.5 # Cognitive
        c2 = 1.5 # Social
        
        for i in range(n_iters):
            if calls >= budget:
                break
                
            # Update inertia
            w = w - (w - w_min) * (i / n_iters)
            
            for p in range(n_particles):
                if calls >= budget:
                    break
                    
                r1 = rng.random(n)
                r2 = rng.random(n)
                
                V[p] = w * V[p] + c1 * r1 * (Pbest_X[p] - X[p]) + c2 * r2 * (Gbest_X - X[p])
                
                # Clamp velocity
                V_max = (ub - lb) * 0.2 # 20% of range
                V[p] = np.clip(V[p], -V_max, V_max)
                
                X[p] = X[p] + V[p]
                X[p] = np.clip(X[p], lb, ub)
                
                F_new = func(X[p])
                calls += 1
                
                if F_new < Pbest_F[p]:
                    Pbest_F[p] = F_new
                    Pbest_X[p] = X[p].copy()
                    
                if F_new < Gbest_F:
                    Gbest_F = F_new
                    Gbest_X = X[p].copy()
                    
        return Gbest_X, Gbest_F
