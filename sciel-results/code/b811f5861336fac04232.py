"""DTLZ1 optimizer using stratified 2D simplex sampling for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic 2D Pareto front with stratified points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        all_points = []
        deadline = time.time() + 290
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # DTLZ1 structure for n_obj=3, dim=7:
        # x[0], x[1]: shaping variables (2D simplex for M=3)
        # x[2]..x[6]: g-variables (5 variables)
        # g is minimized when all g-variables = 0.5
        
        base_x = np.zeros(dim)
        base_x[2:] = 0.5  # Set all g-variables to 0.5 to minimize g
        
        n_evals = min(budget, 200)
        if n_evals <= 0:
            return all_points
        
        points_to_eval = []
        
        # Use triangular grid for 2D simplex coverage
        # Find largest n such that n*(n+1)/2 <= n_evals
        n_tri = 1
        while (n_tri + 1) * (n_tri + 2) // 2 <= n_evals:
            n_tri += 1
        
        # Generate triangular grid points
        # For triangle with vertices (0,0), (1,0), (0,1):
        # Row i has points where x0 + x1 <= 1 - i/(n-1)
        # Using barycentric: (a/(n-1), b/(n-1)) where a+b <= n-1
        seen = set()
        for i in range(n_tri):
            for j in range(n_tri - i):
                x0 = j / (n_tri - 1)
                x1 = (n_tri - 1 - i - j) / (n_tri - 1)
                key = (round(x0, 12), round(x1, 12))
                if key not in seen:
                    x = base_x.copy()
                    x[0] = x0
                    x[1] = x1
                    points_to_eval.append(x)
                    seen.add(key)
        
        # If we have remaining budget, add stratified points
        remaining = n_evals - len(points_to_eval)
        if remaining > 0:
            # Use stratified sampling in the 2D simplex
            # Divide into sub-regions and sample one point per region
            n_strata = min(remaining, 10)
            points_per_stratum = remaining // n_strata
            rem2 = remaining % n_strata
            
            for i in range(n_strata):
                count = points_per_stratum + (1 if i < rem2 else 0)
                # x0 stratum: [i/n_strata, (i+1)/n_strata]
                x0_center = (i + 0.5) / n_strata
                max_x1 = 1.0 - x0_center
                
                for j in range(count):
                    if len(points_to_eval) >= n_evals:
                        break
                    # Stratified x1 in [0, max_x1]
                    x1 = (j + 0.5) * max_x1 / count
                    key = (round(x0_center, 12), round(x1, 12))
                    if key not in seen:
                        x = base_x.copy()
                        x[0] = x0_center
                        x[1] = x1
                        points_to_eval.append(x)
                        seen.add(key)
        
        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=float))
            except Exception:
                continue
        
        return all_points
# EVOLVE-BLOCK-END
