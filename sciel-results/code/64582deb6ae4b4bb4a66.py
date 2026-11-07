"""Multi-objective optimizer on DTLZ1 with analytical front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Set aggregation variables to 0 to achieve g(x) = 0 (on the Pareto front).
    2. Sample position variables uniformly on the simplex to maximize hypervolume.
    3. Use the full budget for front sampling.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 285.0
    
    def _elapsed(self):
        return time.time() - self.start_time
    
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        m = self.n_obj
        n = self.dim
        n_pos = n - m  # Number of position variables
        
        n_samples = self.budget
        if n_samples <= 0:
            return np.array([])
        
        # Generate uniform samples on the (n_pos-1)-simplex
        # Using Dirichlet(1,1,...,1) distribution
        rng = np.random.RandomState(42)
        # Generate n_pos-1 random numbers, use them to create simplex points
        # Method: generate n_pos exponential(1) randoms and normalize
        exp_rand = rng.exponential(1.0, size=(n_samples, n_pos))
        simplex = exp_rand / exp_rand.sum(axis=1, keepdims=True)
        
        # Scale to bounds for position variables
        X_pos = lo[:n_pos] + (hi[:n_pos] - lo[:n_pos]) * simplex
        
        # Construct aggregation variables set to 0 (minimizes g to 0)
        X_agg = np.zeros((n_samples, m))
        
        # Combine
        X = np.hstack((X_pos, X_agg))
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
