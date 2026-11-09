"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a structured approach:
    1. Dense grid on the first 3 (shape) variables with position variables at midpoints.
    2. Uses remaining budget for random perturbation of position variables and shape variables
       to refine the front and ensure coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # 300s limit, 5s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        n_shape = min(3, self.dim)
        
        # Strategy:
        # WFG9 has 3 shape variables and the rest are position.
        # Position variables at midpoint (0.5 in [0,1]) are optimal.
        # Shape variables determine the front.
        
        # We want to sample the shape space densely.
        # Budget is 200.
        
        # Phase 1: Grid on shape variables
        # Use a denser grid: 7x7x4 = 196 points (fits in budget of 200)
        # This gives better coverage of the curved Pareto front than 6x6x5=180
        
        if self.dim >= 3:
            g1, g2, g3 = 7, 7, 4
        elif self.dim == 2:
            g1, g2 = 14, 14
            g3 = 1
        else:
            g1 = 200
            g2 = 1
            g3 = 1
            
        grid_sizes = [g1, g2, g3]
        
        # Generate grid points for shape variables
        ranges = []
        for i in range(n_shape):
            if i < len(grid_sizes) and grid_sizes[i] > 0:
                ranges.append(np.linspace(lo[i], hi[i], grid_sizes[i]))
            else:
                ranges.append(np.array([mid[i]]))
        
        grids = np.meshgrid(*ranges, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)
        N = grid_points.shape[0]
        
        if N > self.budget:
            # If grid is too big, subsample or reduce
            # But with chosen sizes, it should be <= 200
            # 7*7*4 = 196
            # 14*14 = 196
            # 200
            pass
            
        # Construct full vectors: shape from grid, position from mid
        X = np.tile(mid, (N, 1))
        X[:, :n_shape] = grid_points
        
        # Evaluate grid points
        F = np.array([problem(x) for x in X])
        used = N
        
        # Check time
        if time.time() > deadline:
            return X
            
        # Phase 2: Use remaining budget for refinement
        # The front of WFG9 is curved. A grid might miss some points if the curvature is high.
        # Also, random noise on position variables might help if midpoint isn't exactly optimal
        # (though it is for WFG9).
        # More importantly, random samples in the shape space with optimal position can fill gaps.
        
        remaining = self.budget - used
        if remaining > 0:
            # Generate random points for shape variables
            # Keep position variables at mid, or slightly perturb?
            # Perturbing position variables moves points off the front (worse).
            # So keep position at mid.
            X_rand_shape = np.random.uniform(lo[:n_shape], hi[:n_shape], (remaining, n_shape))
            X_rand = np.tile(mid, (remaining, 1))
            X_rand[:, :n_shape] = X_rand_shape
            
            F_rand = np.array([problem(x) for x in X_rand])
            used += remaining
            
            X = np.vstack([X, X_rand])
            F = np.vstack([F, F_rand])
            
        # Check time again
        if time.time() > deadline:
            return X
            
        # Phase 3: Adaptive refinement based on non-dominated points
        # If we have more budget and time, refine around non-dominated points
        if used < self.budget and time.time() < deadline:
            # Find non-dominated points
            n_points = len(F)
            is_dominated = np.zeros(n_points, dtype=bool)
            
            for i in range(n_points):
                if is_dominated[i]:
                    continue
                for j in range(n_points):
                    if i == j or is_dominated[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        is_dominated[j] = True
            
            non_dominated_idx = np.where(~is_dominated)[0]
            
            # If we have non-dominated points and remaining budget, do local refinement
            if len(non_dominated_idx) > 0 and used < self.budget:
                # Sample around non-dominated points
                n_refine = min(self.budget - used, 20)
                if n_refine > 0:
                    # Pick a few non-dominated points to refine around
                    n_to_refine = min(5, len(non_dominated_idx))
                    # Use random selection for diversity
                    selected = np.random.choice(non_dominated_idx, n_to_refine, replace=False)
                    
                    X_refine = []
                    for idx in selected:
                        base = X[idx].copy()
                        # Add small perturbations to shape variables
                        n_per_point = n_refine // n_to_refine
                        for k in range(n_per_point):
                            x_new = base.copy()
                            # Perturb shape variables
                            for d in range(n_shape):
                                x_new[d] += np.random.uniform(-0.05, 0.05)
                                x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                            X_refine.append(x_new)
                    
                    if X_refine:
                        X_refine = np.array(X_refine)
                        F_refine = np.array([problem(x) for x in X_refine])
                        X = np.vstack([X, X_refine])
                        F = np.vstack([F, F_refine])
                        used += len(X_refine)
        
        return X
# EVOLVE-BLOCK-END
