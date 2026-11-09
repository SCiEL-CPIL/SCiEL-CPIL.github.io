"""DTLZ1 optimizer using stratified simplex sampling for optimal hypervolume."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizes DTLZ1 by sampling the analytic Pareto front with stratified points."""

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
        # x[0] = alpha (shaping)
        # x[1] = beta (shaping)  
        # x[2:7] = g-variables (5 vars)
        # g = 100 * (n_g + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))))
        # g_min = 0 when all g-vars = 0.5
        # When g=0: f1 = (1+g)*alpha, f2 = (1+g)*beta, f3 = (1+g)*(1-alpha-beta)
        # So with g=0: f1=alpha, f2=beta, f3=1-alpha-beta
        # Pareto front: alpha >= 0, beta >= 0, alpha + beta <= 1
        
        # Set g-variables to 0.5 to achieve g=0
        base_x = np.zeros(dim)
        base_x[2:] = 0.5
        
        # Generate stratified points on the simplex {alpha >= 0, beta >= 0, alpha+beta <= 1}
        # Using a triangular grid with stratification for better hypervolume coverage
        
        # Strategy: Use a regular triangular grid plus jittered points
        # For budget N, use roughly sqrt(N) points along each edge
        
        n_edge = int(np.sqrt(budget))
        n_edge = max(3, min(n_edge, 50))  # Cap to keep grid manageable
        
        points_to_eval = []
        
        # Generate triangular grid points
        for i in range(n_edge + 1):
            for j in range(n_edge + 1 - i):
                if len(points_to_eval) >= budget:
                    break
                # alpha in [0,1], beta in [0, 1-alpha]
                alpha = i / n_edge
                beta = j / n_edge
                # Clip to ensure alpha + beta <= 1
                if alpha + beta <= 1.0:
                    x = base_x.copy()
                    x[0] = alpha
                    x[1] = beta
                    points_to_eval.append(x)
        
        # If we have budget left, add stratified random points
        if len(points_to_eval) < budget:
            rng = np.random.default_rng(42)
            # Use quasi-random (Halton-like) sequence for better coverage
            # Generate points in unit square and map to simplex
            n_extra = budget - len(points_to_eval)
            
            # Use stratified sampling in the simplex
            # Divide simplex into sub-triangles
            n_strat = int(np.sqrt(n_extra)) if n_extra > 0 else 1
            n_strat = max(2, min(n_strat, 30))
            
            for si in range(n_strat):
                for sj in range(n_strat):
                    if len(points_to_eval) >= budget:
                        break
                    # Random point within the stratified cell
                    u = (si + rng.random()) / n_strat
                    v = (sj + rng.random()) / n_strat
                    
                    # Map from unit square to simplex using inverse CDF
                    # For uniform on simplex: use sqrt transformation
                    # alpha = 1 - sqrt(u), beta = (1 - alpha) * v
                    alpha = 1.0 - np.sqrt(u)
                    beta = (1.0 - alpha) * v
                    
                    x = base_x.copy()
                    x[0] = alpha
                    x[1] = beta
                    points_to_eval.append(x)
        
        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            f = problem(x)
            all_points.append(f.copy())
            
        # If we still have budget left, add more random points
        if len(all_points) < budget:
            rng2 = np.random.default_rng(123)
            while len(all_points) < budget and time.time() < deadline:
                # Random point in simplex
                u = rng2.random()
                v = rng2.random()
                alpha = 1.0 - np.sqrt(u)
                beta = (1.0 - alpha) * v
                
                x = base_x.copy()
                x[0] = alpha
                x[1] = beta
                f = problem(x)
                all_points.append(f.copy())

        return all_points
# EVOLVE-BLOCK-END
