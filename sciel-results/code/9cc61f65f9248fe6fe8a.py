import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG4 specific heuristic:
        # The Pareto front is typically found when the 'distance' variables (x_1...x_{n-1}) are 0.
        # The 'position' variable (x_0) varies.
        # We assume x_0 is the position variable and the rest are distance variables.
        
        # Strategy:
        # 1. Allocate 70% of budget to a dense 1D search on x_0 with x_1...x_{n-1} = lo.
        # 2. Allocate 30% of budget to random search in full space.
        
        n_1d = int(self.budget * 0.7)
        n_rand = self.budget - n_1d
        
        X_1d = np.zeros((n_1d, self.dim))
        # Set distance variables to lower bound
        X_1d[:, 1:] = lo[1:]
        # Distribute position variable x_0 uniformly
        if n_1d > 0:
            X_1d[:, 0] = np.linspace(lo[0], hi[0], n_1d)
            
        # Generate random points
        rng = np.random.default_rng(42)
        X_rand = lo + (hi - lo) * rng.random((n_rand, self.dim))
        
        X = np.vstack([X_1d, X_rand])
        F = np.array([problem(x) for x in X])
        
        # Check deadline
        if time.time() > deadline:
            return X
            
        # Filter non-dominated points
        # A point is non-dominated if no other point is better in all objectives and better in at least one.
        # We want to return the non-dominated set to maximize hypervolume.
        # However, the problem statement says "Every returned objective vector is recorded... return value is ignored".
        # Wait, "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" -> This implies the evaluator records the outputs of `problem(x)` calls?
        # No, "Every returned objective vector is recorded" usually means the evaluator wraps the problem or the optimizer returns a list?
        # Let's re-read carefully: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This phrasing is slightly ambiguous. 
        # "problem(x) ... returns 3 objective values."
        # "Every returned objective vector is recorded" likely refers to the vectors returned by `problem`.
        # If the evaluator records every call to `problem`, then we just need to make 200 calls.
        # The "return value of __call__ is ignored" means we don't need to return the non-dominated set from `__call__`.
        # We just need to ensure we make exactly `budget` calls to `problem` and that they are good.
        
        # So the goal is simply to select 200 points that maximize the hypervolume of the set of their objective vectors.
        
        return X
