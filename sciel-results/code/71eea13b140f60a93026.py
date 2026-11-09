import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        n_evals = 0
        X = []
        F = []
        
        def evaluate(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            fx = np.array(problem(x))
            X.append(x.copy())
            F.append(fx)
            n_evals += 1
            return fx

        # Phase 1: Latin Hypercube Sampling for initial diverse coverage
        n_lhs = min(80, self.budget // 2)
        
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            edges = np.linspace(0, 1, n_lhs + 1)
            points_in_bins = np.random.uniform(edges[:-1], edges[1:])
            np.random.shuffle(points_in_bins)
            lhs_points[:, d] = points_in_bins
        
        lhs_points = lo + lhs_points * span
        
        for i in range(n_lhs):
            if n_evals >= self.budget or time.time() > deadline:
                break
            fx = evaluate(lhs_points[i])
            if fx is None:
                break
        
        if n_evals >= self.budget or time.time() > deadline:
            return X
            
        # Convert to arrays
        X_arr = np.array(X)
        F_arr = np.array(F)
        
        # Helper: find non-dominated points
        def get_nondominated(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated
        
        # Phase 2: Local search with adaptive step size around non-dominated points
        # Use multiple weight vectors for Tchebycheff-like scalarization
        n_weights = 5
        # Create diverse weight vectors on the simplex
        weights = []
        for _ in range(n_weights):
            w = np.random.dirichlet(np.ones(self.n_obj))
            weights.append(w)
        weights = np.array(weights)
        
        # Get initial non-dominated set
        nondom_mask = get_nondominated(F_arr)
        nondom_indices = np.where(nondom_mask)[0]
        
        if len(nondom_indices) == 0:
            nondom_indices = np.arange(len(F_arr))
        
        # Select top K parents by sum of objectives
        nondom_indices = nondom_indices[np.argsort(np.sum(F_arr[nondom_indices], axis=1))]
        k_parents = min(8, len(nondom_indices))
        parent_indices = nondom_indices[:k_parents]
        
        parents = X_arr[parent_indices]
        
        # Phase 3: Iterative local search with decreasing step sizes
        step_sizes = [0.15, 0.08, 0.04, 0.02, 0.01]
        
        for step in step_sizes:
            if n_evals >= self.budget or time.time() > deadline:
                break
            
            # Get current non-dominated points
            F_current = np.array(F)
            X_current = np.array(X)
            nondom_mask = get_nondominated(F_current)
            nondom_idx = np.where(nondom_mask)[0]
            
            if len(nondom_idx) == 0:
                nondom_idx = np.arange(len(F_current))
            
            # Sort by sum and take top
            nondom_idx = nondom_idx[np.argsort(np.sum(F_current[nondom_idx], axis=1))]
            n_to_refine = min(6, len(nondom_idx))
            refine_idx = nondom_idx[:n_to_refine]
            
            for idx in refine_idx:
                if n_evals >= self.budget or time.time() > deadline:
                    break
                
                best_x = X_current[idx].copy()
                best_f = F_current[idx].copy()
                
                # Try multiple perturbations with different weight vectors
                n_attempts = min(3, (self.budget - n_evals) // 2)
                
                for attempt in range(n_attempts):
                    if n_evals >= self.budget or time.time() > deadline:
                        break
                    
                    # Choose a weight vector
                    w = weights[attempt % len(weights)]
                    
                    # Generate perturbation
                    noise = np.random.standard_normal(self.dim)
                    child = best_x + step * span * noise
                    child = np.clip(child, lo, hi)
                    
                    fc = evaluate(child)
                    if fc is None:
                        break
                    
                    # Check if child improves in Tchebycheff sense for this weight
                    # Or simply if it's non-dominated relative to best_f
                    if np.all(fc <= best_f) and np.any(fc < best_f):
                        best_x = child.copy()
                        best_f = fc.copy()
        
        return X
