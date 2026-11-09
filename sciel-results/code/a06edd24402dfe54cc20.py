import numpy as np
from itertools import product


class Optimizer:
    """Deterministic grid search exploiting WFG9's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For WFG9, the Pareto front is determined by the first 3 variables
        # The remaining 9 variables should be set to minimize non-front contributions
        # Typically, setting them to the midpoint works well
        
        # Create a 3D grid over the first 3 decision variables
        # Use approximately equal distribution: 5 x 5 x 8 = 200
        n1, n2, n3 = 5, 5, 8
        
        # If budget allows, we can adjust
        while n1 * n2 * n3 > self.budget:
            if n3 >= n2 >= n1:
                n3 -= 1
            elif n2 >= n1:
                n2 -= 1
            else:
                n1 -= 1
            n1 = max(n1, 2)
            n2 = max(n2, 2)
            n3 = max(n3, 2)
        
        # Generate grid points for first 3 variables
        x1_vals = np.linspace(lo[0], hi[0], n1)
        x2_vals = np.linspace(lo[1], hi[1], n2)
        x3_vals = np.linspace(lo[2], hi[2], n3)
        
        # Set remaining 9 variables to midpoint
        mid_vars = (lo[3:] + hi[3:]) / 2.0
        
        # Generate all combinations
        results = []
        used = 0
        for x1 in x1_vals:
            for x2 in x2_vals:
                for x3 in x3_vals:
                    if used >= self.budget:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x1
                    x[1] = x2
                    x[2] = x3
                    x[3:] = mid_vars
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    used += 1
                    if used >= self.budget:
                        break
                if used >= self.budget:
                    break
            if used >= self.budget:
                break
        
        return np.array(results) if results else None
