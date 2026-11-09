"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using a simple random search strategy.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Simple random search: use all budget evaluations to sample uniformly
        # This is the simplest baseline that uses the full budget.
        # The inspiration 1 (score 1.0) likely used a similar simple approach
        # or a very efficient one. Given the complexity of the previous NSGA-II
        # implementation and its lower score, a simpler, more robust approach
        # that uses all evaluations might perform better.
        
        # Use the global numpy random state seeded by the evaluator
        # The evaluator calls np.random.seed(seed) before creating Optimizer.
        # We should use np.random directly to respect that seed.
        
        # Generate all points at once for efficiency
        n_points = self.budget
        X = lo + (hi - lo) * np.random.random((n_points, dim))
        
        # Evaluate all points
        # We need to be careful about time.
        # If evaluation is slow, we might not finish.
        # But random search is fast to generate.
        
        # We can evaluate in batches to check time, but usually 200 evals is fast.
        # Let's just loop and check time occasionally.
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return None
# EVOLVE-BLOCK-END
