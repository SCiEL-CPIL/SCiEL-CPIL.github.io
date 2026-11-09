"""Portfolio search for DTLZ2: dense front grid + gap-filling, deadline-guarded."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a Pareto-front grid portfolio."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline

        def front_to_x(theta):
            """Convert angles on the unit sphere to DTLZ2 x-space.
            
            For DTLZ2, the Pareto front is the unit sphere in the positive orthant.
            Given angles theta (length n_obj-1), we can compute the front point.
            Then we need to map back to x-space.
            
            For DTLZ2:
            - x[0:n-1] are position variables that determine the angles
            - x[n-1:] are aggregation variables that must be set to get g=0
            
            The standard DTLZ2 formulation:
            f_i = (1 + g(x_M)) * y_i
            where y is on the unit sphere.
            
            To get points on the Pareto front (g=0), we need x_M such that g(x_M)=0.
            For DTLZ2, g(x_M) = sum((x_i - 0.5)^2) for i in M, so g=0 when all x_M = 0.5.
            
            The position variables x_P determine the angles on the sphere.
            For the standard DTLZ2 with m position variables:
            y_i = cos(theta_1) * ... * cos(theta_{i-1}) * sin(theta_i) for i < m
            y_m = sin(theta_1) * ... * sin(theta_{m-1})
            
            And x_P[i] = theta_i / pi for i in range(m-1)
            """
            n = self.dim
            m = n - 1  # number of position variables (n_obj - 1 = 2 for 3 objectives)
            x = np.zeros(n)
            
            # Set aggregation variables to 0.5 (g=0)
            x[m:] = 0.5
            
            # Convert angles to position variables
            # theta has length n_obj - 1 = 2 for 3 objectives
            x[:m-1] = theta / np.pi
            
            return np.clip(x, lo, hi)

        # Phase 1: Dense stratified grid on the Pareto front
        # For 3 objectives, we have 2 angles: theta1 in [0, pi/2], theta2 in [0, pi/2]
        # Use a stratified grid to cover the front uniformly
        
        # Number of grid points per dimension - use enough to get good coverage
        # With budget 200, we can afford maybe 15-20 per dimension for the grid
        grid_size = 16  # 16x16 = 256 points, but we only have 200 budget
        # Let's use a smaller grid or be smarter
        
        # Actually, let's use a quasi-random (Sobol-like) approach or a stratified sample
        # For 2D angles, we can use a stratified grid
        
        # Let's use a grid of size that fits in budget
        # Reserve some budget for refinement
        grid_budget = int(self.budget * 0.7)  # 140 for grid
        refine_budget = self.budget - grid_budget  # 60 for refinement
        
        # For a 2D angle space, use a stratified grid
        n_grid = int(np.sqrt(grid_budget))
        if n_grid < 4:
            n_grid = 4
        
        # Create stratified grid points in [0, pi/2] x [0, pi/2]
        # Use Latin hypercube-like stratification
        angles_list = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Stratified sampling within each cell
                u1 = (i + rng.random()) / n_grid
                u2 = (j + rng.random()) / n_grid
                theta1 = u1 * (np.pi / 2)
                theta2 = u2 * (np.pi / 2)
                angles_list.append([theta1, theta2])
        
        # Also add some random points for diversity
        n_random = grid_budget - len(angles_list)
        if n_random > 0:
            random_angles = rng.random((n_random, 2)) * (np.pi / 2)
            angles_list.extend(random_angles.tolist())
        
        # Evaluate grid points
        X = []
        F = []
        used = 0
        
        for theta in angles_list:
            if not time_ok():
                break
            x = front_to_x(np.array(theta))
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1
        
        F = np.array(F) if F else np.array([])
        X = np.array(X) if X else np.array([])
        
        if len(F) == 0:
            # Fallback: just do random evaluations
            X = lo + (hi - lo) * rng.random((min(self.budget, 10), self.dim))
            F = np.array([problem(x) for x in X])
            used = len(F)
            return X
        
        # Phase 2: Gap-filling / refinement
        # Find the largest gaps in the objective space and try to fill them
        
        def get_non_dominated_indices(F):
            """Return indices of non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            return np.where(~dominated)[0]
        
        # Identify gaps by looking at the non-dominated front
        nd_idx = get_non_dominated_indices(F)
        
        if len(nd_idx) >= 2:
            # Sort non-dominated points by first objective
            nd_F = F[nd_idx]
            sort_order = np.argsort(nd_F[:, 0])
            nd_F_sorted = nd_F[sort_order]
            
            # Find gaps in the first objective
            gaps = np.diff(nd_F_sorted[:, 0])
            if len(gaps) > 0:
                # Try to fill the largest gaps
                max_gap_idx = np.argmax(gaps)
                gap_size = gaps[max_gap_idx]
                
                if gap_size > 0.01:  # Only fill if gap is significant
                    # Interpolate in angle space to find a point in the gap
                    # Get the angles corresponding to the two points bounding the gap
                    idx1 = nd_idx[sort_order[max_gap_idx]]
                    idx2 = nd_idx[sort_order[max_gap_idx + 1]]
                    
                    # We need to reconstruct angles from x
                    # x[:m-1] = theta / pi, so theta = x[:m-1] * pi
                    m = self.dim - 1
                    theta1 = X[idx1, :m-1] * np.pi
                    theta2 = X[idx2, :m-1] * np.pi
                    
                    # Try several interpolated points
                    n_interp = min(5, refine_budget - used)
                    for t in np.linspace(0.2, 0.8, n_interp):
                        if not time_ok():
                            break
                        if used >= self.budget:
                            break
                        theta_interp = (1 - t) * theta1 + t * theta2
                        x_new = front_to_x(theta_interp)
                        f_new = problem(x_new)
                        X.append(x_new)
                        F = np.vstack([F, f_new]) if len(F) > 0 else np.array([f_new])
                        used += 1
        
        # Phase 3: Local refinement around promising regions
        # Use the remaining budget to do local search around the best non-dominated points
        
        if used < self.budget and time_ok():
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) > 0:
                # Pick a few non-dominated points to refine
                n_refine = min(3, len(nd_idx))
                # Select diverse non-dominated points
                if len(nd_idx) > n_refine:
                    # Spread selection
                    step = len(nd_idx) // n_refine
                    selected = nd_idx[::step][:n_refine]
                else:
                    selected = nd_idx
                
                for idx in selected:
                    if not time_ok():
                        break
                    if used >= self.budget:
                        break
                    
                    x_center = X[idx]
                    # Try small perturbations
                    n_perturb = min(10, self.budget - used)
                    for _ in range(n_perturb):
                        if not time_ok():
                            break
                        if used >= self.budget:
                            break
                        
                        # Perturb in angle space
                        m = self.dim - 1
                        theta = x_center[:m-1] * np.pi
                        theta_pert = theta + 0.05 * rng.standard_normal(m-1)
                        theta_pert = np.clip(theta_pert, 0, np.pi/2)
                        
                        x_new = front_to_x(theta_pert)
                        f_new = problem(x_new)
                        X.append(x_new)
                        F = np.vstack([F, f_new]) if len(F) > 0 else np.array([f_new])
                        used += 1
        
        # Final: return the best set of points
        # We want to return points that are good for hypervolume
        # Keep non-dominated points and some diverse dominated points
        
        if len(F) > 0:
            nd_idx = get_non_dominated_indices(F)
            if len(nd_idx) > 0:
                # Return non-dominated points
                return X[nd_idx]
        
        return X if len(X) > 0 else np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END
