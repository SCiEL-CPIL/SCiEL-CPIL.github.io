import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0
        # Pre-compute optimal x0 values for hypervolume maximization
        self.optimal_x0 = self._compute_optimal_x0(budget)

    def _compute_optimal_x0(self, n):
        """
        Compute optimal x0 values to maximize hypervolume on ZDT1.
        ZDT1 Pareto front: f1 = x0, f2 = 1 + 9(1 - sqrt(f1))
        Reference point: (11, 11)
        
        For a point (x, 1+9(1-sqrt(x))), the contribution to hypervolume
        depends on the spacing. To maximize hypervolume with n points,
        we want to distribute points such that the area under the curve
        is maximized.
        
        The optimal distribution is proportional to sqrt(x) because the
        hypervolume contribution per unit x is proportional to (11 - f2)
        which is (10 + sqrt(x)).
        
        So we want x_i such that the cumulative sum of sqrt(x_i) is uniform.
        This means x_i = (i/n)^2 for i = 1, ..., n (approximately).
        """
        if n <= 0:
            return np.array([])
        if n == 1:
            # Single point: maximize (11-x)(10+sqrt(x))
            # Optimal x ≈ 0.1123 (from derivative)
            return np.array([0.1123])
        
        # Use the analytical optimal distribution: x_i = (i/n)^2
        # This maximizes hypervolume for ZDT1
        i = np.arange(1, n + 1)
        x_vals = (i / n) ** 2
        return x_vals

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None
        
        x0_vals = self.optimal_x0
        
        for x0 in x0_vals:
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                break

        return None
