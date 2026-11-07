import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        # Latin Hypercube Sampling for initial coverage
        n_lhs = 50
        rng = np.random.default_rng(42)
        
        # Create LHS points
        lhs_points = np.zeros((n_lhs, n))
        for i in range(n_lhs):
            for j in range(n):
                lhs_points[i, j] = (i + rng.random()) / n_lhs
        
        # Map to bounds
        X_lhs = lo + (hi - lo) * lhs_points
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs
        
        # Check deadline
        if time.time() - start_time > deadline:
            return X_lhs
        
        # Find non-dominated points from LHS
        def is_dominated(f, F):
            # Check if f is dominated by any point in F
            for fj in F:
                if np.all(fj <= f) and np.any(fj < f):
                    return True
            return False
        
        # Get non-dominated set
        non_dom_mask = np.array([not is_dominated(F_lhs[i], F_lhs) for i in range(n_lhs)])
        non_dom_X = X_lhs[non_dom_mask]
        non_dom_F = F_lhs[non_dom_mask]
        
        # If we have budget left, do local refinement around non-dominated points
        if used < budget and len(non_dom_X) > 0:
            # Distribute remaining budget among non-dominated points
            remaining = budget - used
            n_refine = min(len(non_dom_X), 10)  # Refine at most 10 points
            points_per_refine = remaining // n_refine
            if points_per_refine < 1:
                points_per_refine = 1
                n_refine = remaining
            
            # Shuffle and take first n_refine
            indices = rng.permutation(len(non_dom_X))[:n_refine]
            
            for idx in indices:
                center = non_dom_X[idx]
                for _ in range(points_per_refine):
                    if used >= budget:
                        break
                    if time.time() - start_time > deadline:
                        return X_lhs
                    # Local perturbation
                    scale = 0.05 * (hi - lo)
                    perturbation = rng.standard_normal(n) * scale
                    new_point = center + perturbation
                    new_point = np.clip(new_point, lo, hi)
                    F_new = problem(new_point)
                    used += 1
                    # Replace worst point in population if new point is non-dominated
                    # Simple: just add to our list of evaluated points
                    X_lhs = np.vstack([X_lhs, [new_point]])
                    F_lhs = np.vstack([F_lhs, [F_new]])
        
        return X_lhs
