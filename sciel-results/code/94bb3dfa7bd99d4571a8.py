import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DE Parameters
        N = 20
        F = 0.8  # Mutation scale factor
        CR = 1.0 # Crossover rate (1.0 means use mutant directly if selected)
        
        # Initialize Population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F_vals = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Dominance check
        # a dominates b if a is better in all objectives and strictly better in at least one
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        # Main Loop
        while used < self.budget:
            # Generate next generation
            X_new = np.empty_like(X)
            F_new = np.empty_like(F_vals)
            
            for i in range(N):
                if used >= self.budget:
                    break
                
                # Select 3 distinct random parents
                # Ensure they are distinct from each other and ideally i
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                # If any is i, it's okay, but let's avoid if possible for diversity
                # np.random.choice with replace=False ensures distinctness.
                
                # DE/rand/1 mutation
                v = X[r1] + F * (X[r2] - X[r3])
                
                # Binomial Crossover
                # If CR=1.0, we can skip the loop or just take v
                if CR < 1.0:
                    j_rand = np.random.randint(self.dim)
                    mask = np.random.rand(self.dim) < CR
                    mask[j_rand] = True
                    v = np.where(mask, v, X[i])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Evaluate
                f_v = problem(v)
                used += 1
                
                # Selection
                if dominates(f_v, F_vals[i]):
                    X_new[i] = v
                    F_new[i] = f_v
                elif dominates(F_vals[i], f_v):
                    X_new[i] = X[i]
                    F_new[i] = F_vals[i]
                else:
                    # Non-dominated: keep diversity
                    if np.random.rand() < 0.5:
                        X_new[i] = v
                        F_new[i] = f_v
                    else:
                        X_new[i] = X[i]
                        F_new[i] = F_vals[i]
                        
            # Update population
            X = X_new
            F_vals = F_new

        return X
