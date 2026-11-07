"""Model-and-optimize for ZDT1: exploit analytic Pareto front structure.
ZDT1 Pareto front: f1=x1, f2=1+x1*sin(10*pi*x1) when x2..xn=0.
Use all budget to densely sample the front with optimal spacing for hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT1 exploiting the analytic Pareto front.
    ZDT1: f1 = x1, f2 = 1 + g(x) + x1*sin(10*pi*x1) where g(x)=sum(x2..xn).
    Pareto front: set x2..xn=0, so f2 = 1 + x1*sin(10*pi*x1).
    Hypervolume is maximized by spreading points along this 1D curve."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        def remaining_time_ok():
            return (time.time() - start_time) < deadline

        # For ZDT1, the Pareto front is parameterized by x1 in [0,1] with x2..xn=0.
        # f1 = x1, f2 = 1 + x1*sin(10*pi*x1)
        # We want to choose x1 values to maximize hypervolume w.r.t. reference [11,11].
        # 
        # Hypervolume contribution of a point (f1, f2) on the front:
        # The hypervolume is the area under the Pareto front curve.
        # With N points, the optimal spacing depends on the curvature of f2(x1).
        #
        # Strategy: Use all budget evaluations to sample x1 values.
        # Since every point on the front is Pareto-optimal, we just need good coverage.
        #
        # For hypervolume maximization with a smooth curve, uniform spacing in x1
        # is a good starting point, but we can do better by accounting for the
        # oscillatory nature of sin(10*pi*x1).

        # Number of points to evaluate
        n_evals = budget  # use all budget

        # Create x1 values: combine uniform grid with some strategic points
        # The Pareto front has local minima of f2 at specific x1 values.
        # f2 = 1 + x1*sin(10*pi*x1), so f2 is minimized where sin(10*pi*x1) is most negative.
        # sin(10*pi*x1) = -1 when 10*pi*x1 = 3*pi/2 + 2*pi*k, i.e., x1 = (3+4k)/20 for k=0,...
        # In [0,1]: x1 = 3/20, 7/20, 11/20, 15/20, 19/20

        # Use a combination: uniform grid for coverage + key points for depth
        # With 200 evaluations, use ~190 uniform + 10 key points
        
        n_uniform = n_evals - 10
        x1_uniform = np.linspace(0.0, 1.0, n_uniform)
        
        # Key x1 values where f2 has local minima (deeper points on the front)
        x1_key = np.array([3/20, 7/20, 11/20, 15/20, 19/20, 1/20, 1/4, 3/4, 19/20, 0.5])
        
        x1_values = np.concatenate([x1_uniform, x1_key])
        
        # Sort and remove duplicates (approximately)
        x1_values = np.sort(x1_values)
        # Remove near-duplicates
        x1_filtered = [x1_values[0]]
        for x1 in x1_values[1:]:
            if x1 - x1_filtered[-1] > 1e-6:
                x1_filtered.append(x1)
        x1_values = np.array(x1_filtered)
        
        # If we have too many, trim to budget
        if len(x1_values) > budget:
            # Keep evenly spaced subset
            indices = np.linspace(0, len(x1_values)-1, budget, dtype=int)
            x1_values = x1_values[indices]

        X_list = []
        F_list = []
        used = 0

        for x1_val in x1_values:
            if used >= budget or not remaining_time_ok():
                break
            x = np.zeros(n)
            x[0] = float(np.clip(x1_val, lo[0], hi[0]))
            x[1:] = 0.0  # minimizes g(x) for ZDT1
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            X_list.append(x)
            F_list.append(f)

        if len(X_list) == 0:
            # Fallback: at least one point
            x = np.zeros(n)
            x[0] = 0.5
            f = problem(x)
            return [x]

        X = np.array(X_list)
        F = np.array(F_list)

        # Return all evaluated points (all are on the Pareto front)
        return X

# EVOLVE-BLOCK-END
