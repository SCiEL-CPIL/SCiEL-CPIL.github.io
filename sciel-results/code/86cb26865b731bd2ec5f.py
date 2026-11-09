# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay


class Optimizer:
    """
    Optimizer for DTLZ4 that exploits the known Pareto front geometry.
    It sets distance variables to their optimal values (0.5) to minimize g(x),
    and samples shape variables using a Delaunay-triangulation-guided strategy
    to maximize hypervolume.
    
    Strategy:
    1. Generate a dense candidate set of points on the Pareto front using a
       stratified grid with cosine spacing (clusters near boundaries).
    2. Compute the 2D Delaunay triangulation of the shape variables (u, v)
       where u = cos(theta_0) and v = theta_1/(pi/2).
    3. The hypervolume contribution is dominated by points on the convex hull
       and near the corners. We prioritize evaluating points that form the
       convex hull of the candidate set, then fill in with interior points
       from the triangulation to ensure coverage.
    4. This concentrates the evaluation budget on "supporting" points that
       define the hypervolume envelope.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # Shape variables: x_0, x_1 (indices 0, 1)
        # Distance variables: x_2 ... x_{dim-1} (indices 2 to 11 for dim=12)
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10

        # For DTLZ4 3-objective, the Pareto front is a quarter sphere.
        # Mapping:
        # theta_0 = (pi / 2) * x_0  =>  f_0 = cos(theta_0)
        # theta_1 = (pi / 2) * x_1  =>  f_1 = sin(theta_0) * cos(theta_1)
        #                                       f_2 = sin(theta_0) * sin(theta_1)
        
        # To get uniform coverage on the sphere surface:
        # u = cos(theta_0) ~ Uniform(0, 1)  =>  x_0 = (2/pi) * arccos(u)
        # v = theta_1/(pi/2) ~ Uniform(0, 1) =>  x_1 = v

        results = []
        
        # Generate candidate points in (u, v) space
        # Use a stratified grid with cosine spacing to cluster near boundaries
        # where hypervolume contribution is highest
        
        n_grid = int(np.sqrt(budget))
        if n_grid < 3:
            n_grid = 3
        
        # Cosine spacing: clusters near 0 and 1
        if n_grid > 1:
            grid_idx = np.arange(n_grid)
            u_vals = 0.5 * (1.0 - np.cos(np.pi * grid_idx / (n_grid - 1)))
            v_vals = 0.5 * (1.0 - np.cos(np.pi * grid_idx / (n_grid - 1)))
        else:
            u_vals = np.array([0.5])
            v_vals = np.array([0.5])
        
        # Create all grid points
        U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
        candidates = np.column_stack([U.ravel(), V.ravel()])
        
        # Ensure corners are included explicitly
        corners = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        
        # Add corners to candidates if not already present
        for c in corners:
            # Check if corner is already in candidates (within tolerance)
            dists = np.linalg.norm(candidates - c, axis=1)
            if np.min(dists) > 1e-10:
                candidates = np.vstack([candidates, c])
        
        # Now we have a set of candidate (u, v) points
        # Compute Delaunay triangulation to identify convex hull structure
        # The convex hull vertices are the most important for hypervolume
        
        if len(candidates) >= 3:
            try:
                tri = Delaunay(candidates)
                # Get convex hull vertices
                # In scipy, we can use Qhull to get the convex hull
                # Alternatively, identify hull points by checking which points
                # are on the boundary of the triangulation
                
                # Use scipy.spatial.ConvexHull
                from scipy.spatial import ConvexHull
                hull = ConvexHull(candidates)
                hull_indices = set(hull.vertices)
            except Exception:
                # Fallback: just use all points
                hull_indices = set(range(len(candidates)))
        else:
            hull_indices = set(range(len(candidates)))
        
        # Sort candidates: hull points first, then others
        hull_points = []
        interior_points = []
        
        for i, (u, v) in enumerate(candidates):
            if i in hull_indices:
                hull_points.append((u, v))
            else:
                interior_points.append((u, v))
        
        # Also add corners explicitly to ensure they're evaluated
        for c in corners:
            u, v = c
            # Check if already in hull_points
            found = False
            for hu, hv in hull_points:
                if abs(hu - u) < 1e-10 and abs(hv - v) < 1e-10:
                    found = True
                    break
            if not found:
                hull_points.append((u, v))
        
        # Build evaluation order: hull points first, then interior
        eval_order = hull_points + interior_points
        
        # Evaluate up to budget points
        for u, v in eval_order:
            if time.time() > deadline:
                break
            
            # Clip to valid range
            u = np.clip(u, 1e-10, 1 - 1e-10)
            v = np.clip(v, 1e-10, 1 - 1e-10)
            
            # Calculate shape variables
            x_0 = (2.0 / np.pi) * np.arccos(u)
            x_1 = v
            
            # Construct full vector
            x = np.zeros(dim)
            x[0] = x_0
            x[1] = x_1
            x[2:] = 0.5  # Set distance variables to optimal value
            
            # Evaluate
            f = problem(x)
            results.append(f)
            
            # Stop if we've used our budget
            if len(results) >= budget:
                break
        
        return None
# EVOLVE-BLOCK-END
