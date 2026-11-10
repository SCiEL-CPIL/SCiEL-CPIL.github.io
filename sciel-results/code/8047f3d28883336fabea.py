import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover rate
        self.N = 20   # Population size

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = self.N
        if N > self.budget:
            N = self.budget
            
        # Initialize population
        X = np.random.uniform(lo, hi, (N, self.dim))
        F = np.array([problem(x) for x in X])
        used_evals = N
        
        # Ideal point for distance calculation (minimizing, so 0)
        # Reference point is [11, 11], so ideal is [0, 0]
        
        while used_evals < self.budget:
            # Generate offspring for each individual
            for i in range(N):
                if used_evals >= self.budget:
                    break
                
                # Select 3 distinct random parents
                parents = np.random.choice(N, 3, replace=False)
                x_a = X[parents[0]]
                x_b = X[parents[1]]
                x_c = X[parents[2]]
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = x_a + self.F * (x_b - x_c)
                
                # Crossover: Binomial crossover
                u = x_c.copy() # Start with target x_i? No, standard DE uses target x_i
                # Standard DE: u[j] = v[j] if rand < CR or j == j_rand else x_i[j]
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = X[i][j]
                        
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                used_evals += 1
                
                # Selection
                # Check dominance
                # u dominates X[i] if f_u <= F[i] and f_u != F[i]
                # X[i] dominates u if F[i] <= f_u and F[i] != f_u
                
                dom_u_over_i = np.all(f_u <= F[i]) and np.any(f_u < F[i])
                dom_i_over_u = np.all(F[i] <= f_u) and np.any(F[i] < f_u)
                
                if dom_u_over_i:
                    X[i] = u
                    F[i] = f_u
                elif not dom_i_over_u:
                    # Non-dominated
                    # Use distance to ideal point (0,0) as tie-breaker
                    # Smaller distance is better
                    dist_u = np.linalg.norm(f_u)
                    dist_i = np.linalg.norm(F[i])
                    if dist_u < dist_i:
                        X[i] = u
                        F[i] = f_u
                # else: keep X[i]
                
        return X
