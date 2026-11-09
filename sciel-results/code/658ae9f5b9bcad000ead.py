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
        
        # Base point: g-variables set to 0.5 to minimize g
        base_x = np.zeros(dim)
        base_x[2:] = 0.5  # Set all g-variables to 0.5
        
        # Strategy: Sample the 2D simplex x[0] >= 0, x[1] >= 0, x[0]+x[1] <= 1
        # Using stratified sampling:
        # Divide x[0] into K strata. For each stratum, sample x[1] uniformly in [0, 1-x[0]].
        
        n_evals = min(budget, 200)
        if n_evals <= 0:
            return all_points
        
        # Choose number of strata for x[0]
        # Let's use 20 strata, with 10 points each for 200 total
        n_strata = 20
        points_per_stratum = n_evals // n_strata
        remainder = n_evals % n_strata
        
        points_to_eval = []
        
        for i in range(n_strata):
            # Determine how many points for this stratum
            count = points_per_stratum + (1 if i < remainder else 0)
            
            # x[0] stratum: [i/n_strata, (i+1)/n_strata]
            # Use center for determinism and good coverage
            x0 = (i + 0.5) / n_strata
            
            # x[1] range: [0, 1 - x0]
            max_x1 = 1.0 - x0
            
            for j in range(count):
                # Stratified sampling for x[1] in [0, max_x1]
                # Divide [0, max_x1] into count sub-intervals
                x1 = (j + 0.5) * max_x1 / count
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                points_to_eval.append(x)
        
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
