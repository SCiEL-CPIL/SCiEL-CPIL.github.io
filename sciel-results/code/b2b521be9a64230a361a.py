import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        n_evals = 0
        X = []
        F = []
        
        def evaluate(x):
            nonlocal n_evals
            fx = problem(x)
            n_evals += 1
            return fx

        def dominated(a, b):
            """Check if a dominates b (a is better in all objectives and at least one is strictly better)"""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return indices of non-dominated points"""
            n = len(F)
            nondom = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i == j:
                        continue
                    if dominated(F[j], F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nondom.append(i)
            return nondom

        # Phase 1: Grid-based decomposition with LHS per cell
        # Divide budget: 70% for global grid exploration, 30% for local refinement
        n_global = min(140, self.budget)
        n_local = self.budget - n_global
        
        # Create a 2x2x2 grid in the 12D space (8 cells)
        # For each cell, use LHS to sample points
        n_cells = 8
        n_per_cell = n_global // n_cells
        
        # Generate grid centers and local spans
        # For 12 dimensions, we can partition each dimension into 2 parts
        # But that gives 2^12 = 4096 cells, too many
        # Instead, use a coarser partition: group dimensions
        
        # Strategy: Use stratified sampling with 2 levels per dimension
        # This is equivalent to LHS with 2 points per dimension, repeated
        # Actually, let's use a different approach:
        # Generate points in [0,1]^12, then map to bounds
        # Use a grid of 2x2x2 in the first 3 dimensions, and uniform in others
        
        # Better: Use Latin Hypercube with stratification
        # For 12D, use n_global points with LHS
        
        # Generate LHS points for global exploration
        lhs_points = np.zeros((n_global, self.dim))
        for d in range(self.dim):
            edges = np.linspace(0, 1, n_global + 1)
            points_in_bins = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points_in_bins)
            lhs_points[:, d] = points_in_bins
        
        lhs_points = lo + lhs_points * span
        
        # Evaluate LHS points
        for i in range(n_global):
            if n_evals >= self.budget or time.time() > deadline:
                break
            x = lhs_points[i]
            fx = evaluate(x)
            X.append(x)
            F.append(fx)
            
        if n_evals >= self.budget or time.time() > deadline:
            return np.array(X)
            
        X = np.array(X)
        F = np.array(F)
        
        # Phase 2: Local refinement around non-dominated points
        # Find non-dominated points
        nondom = non_dominated_sort(F)
        
        if not nondom:
            nondom = list(range(len(F)))
        
        # Select top K non-dominated points for local search
        # Sort by sum of objectives (lower is better)
        nondom.sort(key=lambda i: np.sum(F[i]))
        k_parents = min(6, len(nondom))
        parents = [X[i] for i in nondom[:k_parents]]
        
        # Use remaining budget for local search
        remaining_budget = self.budget - n_evals
        if remaining_budget > 0 and parents:
            evals_per_parent = max(5, remaining_budget // len(parents))
            
            for parent in parents:
                if time.time() > deadline or n_evals >= self.budget:
                    break
                
                # Local search: perturb parent in random directions
                # Use a small step size that shrinks over iterations
                base_step = 0.1
                for iteration in range(evals_per_parent):
                    if n_evals >= self.budget or time.time() > deadline:
                        break
                    
                    # Random direction in [-1, 1]^12
                    direction = np.random.uniform(-1, 1, self.dim)
                    # Normalize to unit length
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    
                    # Step size decreases with iteration
                    step = base_step * (0.8 ** iteration)
                    x_new = parent + step * direction
                    x_new = np.clip(x_new, lo, hi)
                    
                    fx_new = evaluate(x_new)
                    X = np.vstack([X, x_new])
                    F = np.vstack([F, fx_new])
                    
                    # If this point is non-dominated, use it as new parent
                    if not any(dominated(fx_new, F[j]) for j in range(len(F) - 1)):
                        parent = x_new
                        base_step = 0.05  # Reduce step size after improvement
        
        return np.array(X)
