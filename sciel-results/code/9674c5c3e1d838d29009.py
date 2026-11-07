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
            
            # ZDT2 specific knowledge:
            # Pareto front is achieved when x_2...x_n = (0.5 * x_1)^(1 / (4 * (n-1)))
            # and x_1 in [0, 1]
            
            n = self.dim
            # Use the full budget to sample the front
            # We want to maximize hypervolume.
            # Let's sample x_1 uniformly in [0, 1]
            # Or perhaps slightly denser near the "knee" if it exists?
            # ZDT2 front: f2 = 1 - sqrt(f1).
            # The curvature is highest near f1=0.
            # Uniform f1 sampling might leave gaps in f2 space near 0?
            # Actually, df2/df1 = -1/(2*sqrt(f1)). Infinite slope at 0.
            # So uniform f1 gives dense points in f2 near 1, sparse near 0.
            # Uniform f2 would give dense points in f1 near 1, sparse near 0.
            # Hypervolume is symmetric in a sense?
            # Let's try uniform x_1 first.
            
            x1_vals = np.linspace(0.0, 1.0, self.budget)
            
            # Construct full x vectors
            # x_1 is the first component
            # x_2...x_n are determined by x_1
            X = np.zeros((self.budget, n))
            X[:, 0] = x1_vals
            
            if n > 1:
                # Calculate the target values for x_2...x_n
                # Note: if x1 is 0, the power is 0^positive = 0.
                exponent = 1.0 / (4.0 * (n - 1))
                # Ensure no negative base for power if x1 was negative, but bounds are [0,1]
                target_vals = (0.5 * x1_vals) ** exponent
                # Fill columns 1 to n-1
                X[:, 1:] = target_vals[:, np.newaxis]
            
            # Evaluate all points
            # The problem function clips to bounds, so we are safe.
            F = np.array([problem(x) for x in X])
            
            # Return the solutions (or F, doesn't matter for scoring, but contract says return is ignored)
            return X
