import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Generate K candidate LHS designs
        K = 20 # Number of candidates
        candidates = []
        
        for k in range(K):
            rng = np.random.default_rng(42 + k)
            # Generate LHS
            lhs = np.zeros((budget, dim))
            for d in range(dim):
                # Centered LHS
                centers = (np.arange(budget) + 0.5) / budget
                # Permute to create the LHS structure
                perm = rng.permutation(budget)
                lhs[:, d] = centers[perm]
            candidates.append(lhs)
            
        # Evaluate quality of each candidate
        # Metric: Minimum pairwise distance in [0,1]^dim
        # To speed up, we can use a subsample or just compute for small N=200.
        # 200x200 distance matrix is fast.
        
        best_idx = 0
        best_min_dist = 0.0
        
        for i, lhs in enumerate(candidates):
            # Compute pairwise distances
            # dist2 = sum(a^2) + sum(b^2) - 2ab
            norms = np.sum(lhs**2, axis=1)
            dots = lhs @ lhs.T
            dist2 = norms[:, None] + norms[None, :] - 2 * dots
            dist2 = np.maximum(dist2, 0)
            np.fill_diagonal(dist2, np.inf)
            min_dist = np.sqrt(np.min(dist2))
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
                
        # Select the best candidate
        X_norm = candidates[best_idx]
        
        # Map to bounds
        X = lo + X_norm * span
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
