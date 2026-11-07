"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop = int(min(30, self.budget // 5))
        
        # Use structured initialization for better coverage
        X = np.zeros((pop, self.dim))
        for i in range(pop):
            X[i] = lo + (hi - lo) * (i + 0.5) / pop
        # Add some random perturbation to avoid identical points
        X += (hi - lo) * rng.random((pop, self.dim)) * 0.1
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = pop
        deadline = time.time() + 250  # Safety margin before 330s kill
        
        # Adaptive mutation step size
        step_size = 0.15
        
        while used < self.budget and time.time() < deadline:
            # Find non-dominated points
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            non_dom_idx = np.where(~dominated)[0]
            
            if len(non_dom_idx) > 0:
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            else:
                parent_idx = rng.integers(pop)
            
            parent = X[parent_idx]
            
            # Adaptive mutation
            noise = rng.standard_normal(self.dim)
            child = parent + step_size * (hi - lo) * noise
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the worst point based on crowding distance approximation
            # Use sum of objectives as a simple fitness indicator
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Adapt step size based on whether we found a non-dominated point
            if not np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1)):
                step_size = max(0.05, step_size * 0.95)
            else:
                step_size = min(0.3, step_size * 1.05)
        
        return X
# EVOLVE-BLOCK-END
