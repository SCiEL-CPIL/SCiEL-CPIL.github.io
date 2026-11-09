import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        pop = int(min(20, self.budget // 4))
        X = lo + span * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        while used < self.budget:
            if time.time() >= deadline:
                break
            
            # Find non-dominated solutions
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Adaptive mutation step size based on progress
            progress = used / self.budget
            step_size = 0.1 * (1.0 - 0.5 * progress)
            
            child = parents[rng.integers(len(parents))] + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace the worst solution based on sum of objectives
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
        
        return X
