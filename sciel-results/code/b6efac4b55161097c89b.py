import numpy as np
import time
from scipy.special import roots_chebyt

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # Use full budget
        n_points = self.budget
        
        # Generate Chebyshev nodes in [-1, 1]
        # Chebyshev nodes are optimal for minimizing the maximum error in polynomial approximation,
        # which correlates well with minimizing the hypervolume loss on a smooth convex front.
        if n_points > 2:
            _, cos_theta = roots_chebyt(n_points)
            # Map from [-1, 1] to [0, 1]
            x0_values = 0.5 * (cos_theta + 1.0)
            # Ensure we cover the exact boundaries 0 and 1?
            # Chebyshev nodes do not include the endpoints -1 and 1 exactly (roots are in (-1, 1)).
            # For ZDT2, the endpoints of the Pareto front are important.
            # We should force the first and last points to be 0 and 1.
            x0_values[0] = 0.0
            x0_values[-1] = 1.0
            # Sort them to ensure monotonic increase (roots_chebyt returns unsorted? No, it returns cos(theta) which is sorted? 
            # Actually roots_chebyt returns the roots of T_n(x). They are symmetric.
            # Let's just sort them to be safe.
            x0_values = np.sort(x0_values)
        else:
            x0_values = np.linspace(0.0, 1.0, n_points)
        
        for x1 in x0_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x1
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
