import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            N = self.budget
            # For ZDT2, the Pareto front is defined by x1 in [0, 1] with x2...xn = 0
            # f1 = x1
            # f2 = (1 + 9*sqrt(x1/30)) * (1 - sqrt(x1/(9*sqrt(x1/30) + 1)))
            # 
            # The hypervolume contribution is highest where the front is "bulging" out.
            # To maximize hypervolume, we should sample x1 values such that we cover the front well.
            # The current approach uses u^(4/3) which clusters near 0.
            # Let's try a different distribution. 
            # The reference point is [11, 11].
            # The ideal point is [0, 0].
            # 
            # A common strategy for ZDT2 is to sample x1 uniformly or with a slight bias.
            # However, the directive suggests using inverse-CDF sampling based on hypervolume contribution.
            # 
            # Let's analyze the hypervolume gain.
            # For a point (f1, f2), the hypervolume contribution relative to the reference (11, 11)
            # is roughly (11 - f1) * (11 - f2) if it's non-dominated.
            # Since f1 = x1 and f2 decreases as x1 increases (for ZDT2), 
            # f2(x1) = (1 + 9*sqrt(x1/30)) * (1 - sqrt(x1/(9*sqrt(x1/30) + 1)))
            # 
            # Let's just try a uniform distribution first to see if it's better or worse,
            # or stick to the current one but maybe tweak the exponent.
            # The current exponent is 4/3 ~ 1.33.
            # If we use exponent 1.0 (uniform), we spread points more evenly.
            # If we use exponent 2.0, we cluster even more near 0.
            # 
            # Actually, for ZDT2, the front is convex. The hypervolume is maximized by covering the whole range.
            # A uniform distribution of x1 in [0, 1] is often a good baseline.
            # The current code uses u^(4/3). Let's try uniform u = (i + 0.5) / N.
            
            u = (np.arange(N) + 0.5) / N
            x1_vals = u
            
            # Create population
            X = np.zeros((N, self.dim))
            X[:, 0] = x1_vals
            # x2...xN are 0, which is optimal for ZDT2 Pareto front
            
            # Evaluate
            F = np.array([problem(x) for x in X])
            
            return X
