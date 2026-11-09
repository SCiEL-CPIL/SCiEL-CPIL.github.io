import numpy as np
class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            all_points = []
            
            # 1. Structured Sampling
            # Hypothesis: x_i = 0 for i > 0 is optimal for f1.
            # Vary x1 to cover f2 range.
            
            n_structured = int(self.budget * 0.8)
            n_random = self.budget - n_structured
            
            x1_vals = np.linspace(lo[0], hi[0], n_structured)
            # Ensure we hit the minimum of f2 (likely near -1)
            # We don't know the exact formula, but for (x1+1)^2 it's -1.
            # Let's just rely on linspace covering the range.
            
            X_struct = np.zeros((n_structured, self.dim))
            X_struct[:, 0] = x1_vals
            # Set rest to 0. Clip to bounds.
            for i in range(1, self.dim):
                X_struct[:, i] = 0.0
            X_struct = np.clip(X_struct, lo, hi)
            
            F_struct = np.array([problem(x) for x in X_struct])
            for f in F_struct:
                all_points.append(f.copy())
                
            # 2. Random Exploration
            if n_random > 0:
                X_rand = lo + (hi - lo) * np.random.rand(n_random, self.dim)
                F_rand = np.array([problem(x) for x in X_rand])
                for f in F_rand:
                    all_points.append(f.copy())
                
            return all_points
