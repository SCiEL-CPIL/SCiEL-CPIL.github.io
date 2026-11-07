import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Number of subproblems
        M = 20
        # Population size is M
        # Initialize weights using simplex lattice or similar
        # For 3 objectives, we can generate weights on a grid
        # Weights sum to 1
        weights = []
        # Simplex lattice generation for n_obj dimensions
        # Number of points on grid with level k
        # Let's use a fixed set of weights for simplicity and robustness
        # For 3 objectives, we can use a 5x5 grid on the simplex
        # w1 + w2 + w3 = 1, wi >= 0
        k = 4 # grid resolution
        for i in range(k + 1):
            for j in range(k + 1 - i):
                w1 = i / k
                w2 = j / k
                w3 = 1 - w1 - w2
                if w3 >= -1e-9:
                    weights.append([w1, w2, max(0, w3)])
        
        # Ensure we have exactly M weights, or adjust M
        # The above generates (k+1)(k+2)/2 points. For k=4, 15 points.
        # Let's set M = len(weights)
        M = len(weights)
        weights = np.array(weights)
        
        # If M is too small, increase k or use a different strategy
        # If M > budget, reduce M
        if M > self.budget // 5:
            # Reduce M by subsampling
            indices = np.random.choice(M, self.budget // 5, replace=False)
            weights = weights[indices]
            M = len(weights)
        
        # Initialize population
        X = np.zeros((M, self.dim))
        F = np.zeros((M, self.n_obj))
        
        for i in range(M):
            X[i] = lo + (hi - lo) * np.random.random(self.dim)
            F[i] = problem(X[i])
        
        used = M
        
        # Define neighborhood
        # For each i, find the M//2 closest weights
        # Distance in weight space
        dists = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        # Sort distances and pick top M_nb
        M_nb = max(2, M // 2)
        neighbors = []
        for i in range(M):
            sorted_idx = np.argsort(dists[i])
            neighbors.append(sorted_idx[:M_nb])
        
        # Evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a random subproblem
            i = np.random.randint(M)
            
            # Select parents from neighborhood of i
            parent_indices = neighbors[i]
            p1_idx = parent_indices[np.random.randint(len(parent_indices))]
            p2_idx = parent_indices[np.random.randint(len(parent_indices))]
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            
            # Crossover (SBX)
            # Simple blend crossover for simplicity and speed
            alpha = np.random.random()
            child = alpha * p1 + (1 - alpha) * p2
            
            # Mutation (Gaussian)
            sigma = 0.1
            child = child + sigma * (hi - lo) * np.random.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update neighborhood
            # If child dominates any solution in the neighborhood, replace it
            # Check against all in neighborhood
            for j in parent_indices:
                if np.all(child <= X[j]) and np.any(child < X[j]):
                    X[j] = child
                    F[j] = fc
                    break # Only replace one to save computation? Or replace all dominated?
                    # Standard MOEA/D replaces if it dominates the current solution in that subproblem's slot
                    # But here we are using a neighborhood list.
                    # Let's replace the worst one in the neighborhood if child is non-dominated
                    # Or simply: if child dominates X[j], replace X[j].
        
        # Return the non-dominated set
        # Extract non-dominated points from X, F
        nd_mask = np.zeros(M, dtype=bool)
        for i in range(M):
            dominated = False
            for j in range(M):
                if i != j:
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
            if not dominated:
                nd_mask[i] = True
        
        # The return value is ignored, but we must not crash
        return X[nd_mask]
