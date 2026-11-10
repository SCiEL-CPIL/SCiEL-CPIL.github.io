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
        
        # Generate a standard LHS
        rng = np.random.default_rng(42)
        X = np.zeros((budget, dim))
        for d in range(dim):
            intervals = np.arange(budget) / budget
            offsets = rng.random(budget)
            X[:, d] = intervals + offsets * (1.0 / budget)
            X[:, d] = X[rng.permutation(budget), d]
            
        # Map to bounds
        X = lo + X * span
        
        # Optimize the LHS to maximize minimum distance
        # Use a simple iterative improvement
        # Compute initial distances
        norms = np.sum(X**2, axis=1)
        dots = X @ X.T
        dist2 = norms[:, None] + norms[None, :] - 2 * dots
        dist2 = np.maximum(dist2, 0)
        
        # We want to increase the minimum distance.
        # Strategy: Find the point with the smallest minimum distance to others.
        # Try to move it or swap it?
        # Swapping with a point from a larger pool is better.
        
        # Let's stick to the current Maximin-from-pool but improve the pool generation.
        # The current pool is 2000 points.
        # Let's try 5000 points.
        # And use a better LHS generation for the pool?
        # Actually, the current code's Maximin is the main strength.
        
        # Let's try Sobol. It's a "deterministic uniform design".
        # I will use Sobol.
        
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=False)
        m = int(np.ceil(np.log2(budget)))
        X_sobol = sampler.random(2**m)
        X = X_sobol[:budget]
        
        # Map to bounds
        X = lo + X * span
        
        F = np.array([problem(x) for x in X])
        
        return X
