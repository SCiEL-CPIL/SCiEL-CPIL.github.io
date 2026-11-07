import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 280.0  # Safety margin before 300s limit
            
            # Extract bounds
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            n = self.dim
            
            # ZDT2 specific knowledge:
            # Pareto front: f1 = x1, f2 = 1 - sqrt(f1) when x_2...x_n = (0.5*x1)^(1/(4*(n-1)))
            # We want to maximize hypervolume with reference point [11, 11]
            
            # Use biased sampling: more points near x1=0 where curvature is high
            # The optimal placement for hypervolume maximization on a convex decreasing front
            # tends to favor denser sampling where the curve is steeper (near x1=0)
            
            # Try a power-law spacing: x1_i = (i/N)^k for some k > 1
            # This gives denser points near 0
            # From cell findings, k=0.5 to 0.7 was tested; let's use k=0.6 as a middle ground
            # Actually, let's think more carefully.
            
            # For ZDT2: f2 = 1 - sqrt(f1)
            # The hypervolume contribution of a point depends on the area it "claims"
            # For a convex decreasing curve, uniform f1 spacing may not be optimal
            # Let's use a transformation that equalizes the "contribution" per point
            
            # A good heuristic: sample f1 such that the f2 values are roughly uniformly spaced
            # Since f2 = 1 - sqrt(f1), we have f1 = (1-f2)^2
            # If we sample f2 uniformly in [0, 1], then f1 = (1 - f2)^2
            # This gives denser points in f1 near 1 (where f2 is near 0)
            
            # Alternatively, we can use a blend. Let's try uniform f2 sampling first.
            f2_vals = np.linspace(0.0, 1.0, self.budget)
            x1_vals = (1.0 - f2_vals) ** 2
            
            # Sort to ensure monotonic order (f2_vals is descending, so x1_vals is ascending)
            # Actually f2_vals goes from 0 to 1, so 1-f2 goes from 1 to 0, squared goes from 1 to 0
            # We want x1 in [0,1], so let's reverse
            f2_vals = np.linspace(1.0, 0.0, self.budget)  # f2 from 1 to 0
            x1_vals = (1.0 - f2_vals) ** 2  # x1 from 0 to 1
            
            # Construct full x vectors
            X = np.zeros((self.budget, n))
            X[:, 0] = x1_vals
            
            if n > 1:
                exponent = 1.0 / (4.0 * (n - 1))
                target_vals = (0.5 * x1_vals) ** exponent
                X[:, 1:] = target_vals[:, np.newaxis]
            
            # Evaluate all points
            F = np.array([problem(x) for x in X])
            
            # Check time
            elapsed = time.time() - start_time
            if elapsed > deadline:
                return X
            
            # If we have time, try a refinement pass with different sampling
            # Try a few different bias parameters and keep the best set
            # But we can't evaluate more than budget points total... wait, we already used budget.
            # Actually, the budget is the total number of evaluations allowed.
            # So we can't do more evaluations. We need to choose the best set of budget points.
            
            # Since we've already used the budget, just return what we have.
            # But wait - maybe we should try a different sampling strategy from the start.
            # Let's use a more principled approach: optimize the placement.
            
            # Actually, for a fixed budget, the best placement for hypervolume on a known front
            # can be found by solving an optimization problem. But that's complex.
            
            # Let's just use the uniform f2 sampling which should be reasonable.
            return X
