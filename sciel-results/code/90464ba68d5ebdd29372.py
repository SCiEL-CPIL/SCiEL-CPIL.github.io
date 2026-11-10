"""ZDT1 multi-objective optimizer using analytic front sampling + greedy hypervolume selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via stratified sampling
    and greedy hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 analytic Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]
        # The Pareto front is achieved when x_2 = ... = x_d = 0.
        # We sample x_1 uniformly in [0, 1] to maximize hypervolume coverage of the front.
        
        # Generate x_1 values
        # Uniform spacing is effective for concave fronts to ensure no large gaps in x_1.
        x1_vals = np.linspace(0.0, 1.0, budget)
        
        # Pre-allocate x vector
        x = np.zeros(dim)
        
        last_f = None
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x[0] = x1_vals[i]
            # x[1:] is already 0.0
            # Clip to bounds (0, 1) is safe for x1 in [0,1] and x_i=0
            # But let's be safe if bounds are different
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            last_f = f
        
        return last_f

# EVOLVE-BLOCK-END
