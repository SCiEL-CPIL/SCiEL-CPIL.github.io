# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest
    triangle area to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2)
    """
    n = 13
    start_time = time.time()
    timeout = 600  # seconds, keep margin from 630

    # Precompute triangle indices for vectorized area calculation
    # C(13, 3) = 286 triangles
    tri_indices = list(itertools.combinations(range(n), 3))
    tri_idx = np.array(tri_indices) # shape (286, 3)
    
    # Function to compute the metric vectorized
    def compute_metric(points):
        # points shape (n, 2)
        # Extract coordinates for all triangles
        # tri_idx shape (M, 3) where M=286
        # p1 = points[tri_idx[:, 0]], etc.
        p1 = points[tri_idx[:, 0]]
        p2 = points[tri_idx[:, 1]]
        p3 = points[tri_idx[:, 2]]
        
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        # Vectorized:
        # cross product magnitude in 2D
        # Area = 0.5 * | (p2-p1) x (p3-p1) |
        v1 = p2 - p1
        v2 = p3 - p1
        # 2D cross product: v1_x * v2_y - v1_y * v2_x
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        # Calculate hull area
        try:
            hull = scipy.spatial.ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return -1e9
            
        if hull_area < 1e-10:
            return -1e9
            
        return min_area / hull_area

    def objective(points_flat):
        points = points_flat.reshape(n, 2)
        val = compute_metric(points)
        if val < -1e8:
            return 1e9 # Large penalty
        return -val

    # Initial guess: Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_points = init_points.copy()
    best_score = compute_metric(best_points)
    
    # Helper to run local optimization from a starting point
    def local_optimize(start_pts):
        try:
            res = scipy.optimize.minimize(
                objective,
                start_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            return res.x.reshape(n, 2)
        except Exception:
            return start_pts

    # Simulated Annealing
    current_points = init_points.copy()
    current_score = best_score
    
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    # We will iterate for a fixed number of steps or until time limit
    # Each step: Local Optimize -> SA Perturbation -> Accept/Reject
    # Local Optimize is expensive, so we do it every few SA steps or adaptively.
    
    steps = 0
    max_steps = 5000 # Safety cap on loop iterations
    
    while time.time() - start_time < timeout - 10:
        # 1. Local Refinement
        # Run Nelder-Mead to find local optimum
        current_points = local_optimize(current_points)
        current_score = compute_metric(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
        # 2. SA Perturbation
        # Generate a random perturbation
        # Scale perturbation with temperature
        scale = 0.1 * T + 0.01
        perturbation = np.random.normal(0, scale, (n, 2))
        proposed_points = current_points + perturbation
        
        proposed_score = compute_metric(proposed_points)
        
        # Acceptance criterion
        delta = proposed_score - current_score
        if delta > 0:
            # Improvement
            current_points = proposed_points
            current_score = proposed_score
        else:
            # Worse, accept with probability exp(delta/T)
            if np.random.random() < np.exp(delta / T):
                current_points = proposed_points
                current_score = proposed_score
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        steps += 1
        
        # Periodic restart if stuck?
        # If no improvement for a long time, maybe restart from best?
        # For now, rely on SA to explore.

    # Final check: ensure best_points is valid
    if compute_metric(best_points) < 0:
        # Fallback to initial
        best_points = init_points

    return best_points

# EVOLVE-BLOCK-END
