# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit per evaluation, keep margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # 1. Latin Hypercube Sampling for initial broad coverage
        # We use a fraction of the budget for LHS to ensure diversity
        n_lhs = min(50, self.budget // 4)
        
        # Generate LHS points
        # For each dimension, divide the range into n_lhs intervals and pick one point in each
        lxs = np.zeros((n_lhs, n))
        for i in range(n):
            # Randomly permute the intervals
            perm = np.random.permutation(n_lhs)
            # Pick a point in each interval: (perm[j] + random) / n_lhs
            # This maps to [0, 1)
            lxs[:, i] = (perm + np.random.random(n_lhs)) / n_lhs
            
        # Scale to bounds
        X_lhs = lo + lxs * (hi - lo)
        
        # Evaluate LHS points
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs
        
        # Combine X and F
        X = X_lhs
        F = F_lhs
        
        # 2. Local Search Refinement
        # Identify the best points (non-dominated or simply best in sum) to refine
        # We will use Nelder-Mead on a subset of the best points
        # To keep it simple and robust, we pick the top K points based on a scalarization
        # or just the non-dominated front.
        
        # Find non-dominated points
        def is_dominated(f, F_all):
            # f is dominated if there exists f' in F_all such that f' <= f and f' != f
            # Vectorized check
            # f' <= f for all objectives
            le = np.all(F_all <= f, axis=1)
            # f' != f (strictly better in at least one)
            lt = np.any(F_all < f, axis=1)
            return np.any(le & lt)

        # We will perform local search on the best few points
        # Sort by sum of objectives to find "best" candidates
        sums = np.sum(F, axis=1)
        # Get indices of the best 5 points
        k_refine = min(5, len(X))
        best_indices = np.argsort(sums)[:k_refine]
        
        # We will use the remaining budget for local search
        # Nelder-Mead can be expensive, so we limit the number of iterations per point
        # or use a simpler gradient-free method.
        # Since we have a strict budget of 200 evaluations, and we used ~50, we have ~150 left.
        # Nelder-Mead in 12D might take many evaluations.
        # Let's use a simpler local search: coordinate descent or random perturbation around best points.
        # Given the "STAGNATION" warning, a structural change like LHS is good, but we need to ensure we find the front.
        # WFG4 has a specific structure.
        
        # Let's try a simple local search: for each of the best points, do a few steps of Nelder-Mead
        # or just random walk with decreasing step size.
        # Actually, let's stick to the LHS + a simple local refinement using the remaining budget.
        
        # We will distribute the remaining budget among the best points.
        remaining_budget = self.budget - used
        
        if remaining_budget > 0:
            # Simple local search: perturb the best points
            # We'll use a simple random walk with adaptive step size
            step_size = 0.1
            for _ in range(remaining_budget):
                if time.time() > deadline - 5:
                    break
                    
                # Pick one of the best points to perturb
                # Re-evaluate which are best? Or just stick to the initial best?
                # Let's re-evaluate the non-dominated front periodically? No, too expensive.
                # Just perturb the current best sum points.
                # Actually, let's just do a random walk on the current population's best points.
                
                # Select a parent from the current best (based on sum)
                sums = np.sum(F, axis=1)
                best_idx = np.argmin(sums)
                parent = X[best_idx]
                
                # Perturb
                child = parent + step_size * (hi - lo) * np.random.standard_normal(n)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace the worst point in the population
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = child
                F[worst_idx] = fc
                
                # Decrease step size slowly
                step_size *= 0.99
                
        return X
# EVOLVE-BLOCK-END
