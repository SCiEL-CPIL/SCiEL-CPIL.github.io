# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Optimized path for moving a sofa through an L-shaped corridor.
    Uses gradient-based optimization on a parametric path (Bezier-like) to maximize
    the intersection area of the corridor placements.
    """
    start_time = time.time()
    deadline = 630.0  # seconds
    safety_margin = 30.0  # seconds to spare

    # Sofa grid points
    # i = 0..449 -> x from -3 to 1.49
    # j = 0..299 -> y from -1.5 to 1.49
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    sofa_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape (N, 2)
    N = sofa_points.shape[0]

    # Base corridor definition:
    # Union of:
    #   A: {0 <= u <= 1, -3 <= v <= 1}
    #   B: {-3 <= u <= 1, 0 <= v <= 1}
    # A point (u, v) is inside if (0<=u<=1 and -3<=v<=1) or (-3<=u<=1 and 0<=v<=1)

    def in_base(u, v):
        return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))

    def corridor_mask(px, py, angle_deg):
        """
        Returns boolean array of shape (N,) indicating which sofa points are inside
        the corridor placement at (px, py) rotated by angle_deg.
        """
        a = np.deg2rad(angle_deg)
        c = np.cos(a)
        s = np.sin(a)
        # R(-a) = [[c, s], [-s, c]]
        dx = sofa_points[:, 0] - px
        dy = sofa_points[:, 1] - py
        u = c * dx + s * dy
        v = -s * dx + c * dy
        return in_base(u, v)

    def evaluate_path(path):
        """
        path: array of shape (n, 3) with rows (x, y, angle_deg).
        Returns the number of sofa points inside ALL placements (including endpoints).
        """
        # Full path with endpoints
        full_path = np.vstack([
            np.array([0.0, 0.0, 0.0]),
            path,
            np.array([-2.0, 0.0, 90.0])
        ])
        
        # Densify: between consecutive rows, add linear interpolations
        # m = max(1, ceil(|a2 - a1|)) equal steps
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        
        densified = np.array(densified)
        
        # Compute intersection: point must be inside ALL placements
        mask = np.ones(N, dtype=bool)
        for i in range(densified.shape[0]):
            px, py, ang = densified[i]
            m = corridor_mask(px, py, ang)
            mask &= m
            if not np.any(mask):
                break
        return np.sum(mask)

    # Parametric path: use a few control points
    # We'll optimize the positions and angles of intermediate waypoints.
    # Start with a reasonable initial guess (the baseline path).
    
    # Baseline path
    baseline = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ])
    
    best_path = baseline.copy()
    best_score = evaluate_path(best_path)
    
    # Try to improve using local search / optimization
    # We'll use a simple coordinate ascent or Nelder-Mead style approach
    # since the objective is non-smooth (count of points).
    
    # Use a simple local search: perturb each waypoint slightly and keep if better
    n_waypoints = best_path.shape[0]
    step_size = 0.05
    max_iter = 500
    
    for iteration in range(max_iter):
        if time.time() > deadline - safety_margin:
            break
        
        improved = False
        for i in range(n_waypoints):
            for dim in range(3):
                for sign in [1, -1]:
                    candidate = best_path.copy()
                    candidate[i, dim] += sign * step_size
                    score = evaluate_path(candidate)
                    if score > best_score:
                        best_path = candidate
                        best_score = score
                        improved = True
        
        if not improved:
            # Try smaller steps
            step_size *= 0.5
            if step_size < 0.001:
                break
    
    return best_path

# EVOLVE-BLOCK-END
