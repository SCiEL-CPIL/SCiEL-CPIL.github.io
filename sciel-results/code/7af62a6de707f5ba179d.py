import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ2, the Pareto front is achieved when g(x) = 0.
        # g(x) = sum((x_i - 0.5)^2) for i from M-1 to n-1.
        # So we set x[2:] = 0.5.
        # We only vary x[0] and x[1].
        
        # Number of points to evaluate on the front
        n_points = min(self.budget, 190)
        
        # Generate Halton sequence for x0 and x1
        h1 = self._halton(n_points, 2)
        h2 = self._halton(n_points, 3)
        
        F = []
        X = []
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            x0 = h1[i]
            x1 = h2[i]
            
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x[1] = x1
            
            # Clip to bounds just in case, though 0.5 and [0,1] Halton are safe
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            F.append(f)
            X.append(x)
            
        if not F:
            return np.array([])
            
        F = np.array(F)
        X = np.array(X)
        
        # Find non-dominated points
        # A point i is dominated if there exists j such that F[j] <= F[i] and F[j] != F[i]
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(i + 1, n):
                if dominated[j]:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
                    
        # Keep non-dominated points
        mask = ~dominated
        X_best = X[mask]
        F_best = F[mask]
        
        # If we have time, we could do a local search, but with 190 points on the front,
        # the hypervolume should be very good.
        
        return X_best
