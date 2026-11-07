import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random
        
        n_starts = 18
        evals_per_local = 8
        
        total_evals = 0
        # Store (x, f) pairs to avoid re-evaluating
        points = []
        
        # Generate weight vectors for scalarization
        weights = []
        for i in range(n_starts):
            w = rng.dirichlet(np.ones(self.n_obj))
            weights.append(w)
        
        indices = list(range(n_starts))
        rng.shuffle(indices)
        
        for i in indices:
            if total_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Initial point
            x0 = lo + (hi - lo) * rng.random(self.dim)
            f0 = problem(x0)
            total_evals += 1
            points.append((x0, f0))
            
            # Local search
            w = weights[i]
            best_x = x0
            best_f = f0
            best_weighted = np.dot(w, best_f)
            
            for _ in range(evals_per_local):
                if total_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Random perturbation
                scale = 0.1
                step = scale * (hi - lo) * rng.standard_normal(self.dim)
                x_new = np.clip(best_x + step, lo, hi)
                
                # Check if we've seen this point before? Unlikely with floats.
                f_new = problem(x_new)
                total_evals += 1
                points.append((x_new, f_new))
                
                weighted_new = np.dot(w, f_new)
                
                if weighted_new < best_weighted:
                    best_x = x_new
                    best_f = f_new
                    best_weighted = weighted_new
        
        if not points:
            return np.array([])
        
        X = np.array([p[0] for p in points])
        F = np.array([p[1] for p in points])
        
        # Non-dominated filter
        n_points = len(F)
        dominated = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            if dominated[i]:
                continue
            for j in range(i + 1, n_points):
                if dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        return X[~dominated]
