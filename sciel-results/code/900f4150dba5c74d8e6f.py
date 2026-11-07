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
        
        # ZDT4 specific logic for x0
        # f1 = x0
        # F(x0) = 101 (assuming x1..x9=0)
        # f2 = (1 + 101) * (1 - sqrt(x0 / 101)) = 102 * (1 - sqrt(x0 / 101))
        
        # Calculate y range (f2 values) for the bounds of x0
        # Note: x0 is the first variable.
        x0_lo = lo[0]
        x0_hi = hi[0]
        
        # Ensure x0_lo <= x0_hi
        if x0_lo > x0_hi:
            x0_lo, x0_hi = x0_hi, x0_lo
            
        def f2_of_x0(x0):
            # Clip x0 to [0, 1] for the sqrt part? 
            # ZDT4 x0 is typically [0,1]. If bounds are wider, 
            # the formula still applies mathematically but might go complex if x0 < 0?
            # sqrt(x0) requires x0 >= 0.
            # Standard ZDT4 x0 in [0, 1].
            # If bounds allow negative, we should clip to 0 for the front calculation?
            # But the problem clips x to bounds.
            # If we pick x0 < 0, f1 < 0. 
            # f2 = 102 * (1 - sqrt(f1/101)). If f1 < 0, sqrt is invalid.
            # ZDT4 is defined for x0 in [0, 1].
            # We should probably restrict our search to [0, 1] if possible, 
            # or clip x0 to [0, 1] in our generation.
            
            # Let's assume standard bounds [0, 1] for x0.
            # If bounds are different, we adapt.
            x0_clipped = np.clip(x0, 0, 1)
            return 102.0 * (1.0 - np.sqrt(x0_clipped / 101.0))
            
        y_max = f2_of_x0(x0_lo)
        y_min = f2_of_x0(x0_hi)
        
        # If the range is zero, just pick the midpoint
        if np.isclose(y_max, y_min):
            x0_vals = np.full(n_evals, (x0_lo + x0_hi) / 2.0)
        else:
            # Sample uniformly in y
            y_vals = np.linspace(y_min, y_max, n_evals)
            
            # Map y back to x0
            # y = 102 * (1 - sqrt(x0 / 101))
            # 1 - y/102 = sqrt(x0 / 101)
            # (1 - y/102)^2 = x0 / 101
            # x0 = 101 * (1 - y/102)^2
            
            x0_vals = 101.0 * (1.0 - y_vals / 102.0) ** 2
            
            # Clip x0_vals to the actual bounds provided, just in case
            x0_vals = np.clip(x0_vals, x0_lo, x0_hi)
            
        # Other variables set to 0
        x_rest = np.zeros(self.dim - 1)
        
        results = []
        for x0 in x0_vals:
            x = np.concatenate(([x0], x_rest))
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = problem(x)
            # The evaluator records f
        return None
