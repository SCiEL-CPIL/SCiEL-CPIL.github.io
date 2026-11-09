import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Small population to allow more generations with tight budget
        N = 10
        # DE Parameters
        F_diff = 0.8
        CR = 0.9
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Main loop
        while used < self.budget:
            for i in range(N):
                if used >= self.budget:
                    break
                
                # Select 3 distinct parents a, b, c from 0..N-1, not i
                candidates = [k for k in range(N) if k != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)
                
                # Mutation
                v = X[a] + F_diff * (X[b] - X[c])
                
                # Crossover
                u = X[i].copy()
                j_rand = rng.integers(0, self.dim)
                
                # Vectorized crossover for speed
                mask = rng.random(self.dim) < CR
                mask[j_rand] = True
                u = np.where(mask, v, X[i])
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                used += 1
                
                # Selection
                f_i = F[i]
                
                # Check dominance
                u_dom_i = np.all(f_u <= f_i) and np.any(f_u < f_i)
                i_dom_u = np.all(f_i <= f_u) and np.any(f_i < f_u)
                
                if u_dom_i:
                    X[i] = u
                    F[i] = f_u
                elif i_dom_u:
                    pass
                else:
                    # Non-dominated
                    # Compare dominance counts in the current population
                    count_u = 0
                    count_i = 0
                    for k in range(N):
                        if k == i:
                            continue
                        f_k = F[k]
                        if np.all(f_u <= f_k) and np.any(f_u < f_k):
                            count_u += 1
                        if np.all(f_i <= f_k) and np.any(f_i < f_k):
                            count_i += 1
                    
                    if count_u > count_i:
                        X[i] = u
                        F[i] = f_u
                    elif count_u < count_i:
                        pass
                    else:
                        if rng.random() < 0.5:
                            X[i] = u
                            F[i] = f_u
        
        return X
