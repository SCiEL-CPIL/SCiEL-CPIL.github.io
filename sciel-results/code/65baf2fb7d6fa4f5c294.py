# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _generate_hexagonal_16():
    """
    Generate a hexagonal packing configuration.
    Center (1) + Ring 1 (6) + Ring 2 (9) = 16.
    """
    points = [[0.0, 0.0]]
    r1 = 1.0
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Need 9 more.
    # Use actual hex lattice points closest to origin.
    hex_pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Sort by distance to origin
    dists = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists)
    # Take first 16
    pts = hex_pts[idx[:16]]
    return pts

def _generate_concentric_rings_16():
    """
    Generate 16 points as two concentric regular octagons.
    Outer radius 1.0, Inner radius ~0.5 (tuned for optimal ratio).
    """
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r_outer = 1.0
    r_inner = 0.5
    pts1 = np.vstack([
        np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)]),
        np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    ])
    return pts1

def _generate_random_16(rng):
    """Generate 16 random points in a unit square."""
    return rng.uniform(-1, 1, (16, 2))

def _reduce_symmetry(points):
    """
    Fix point 0 at origin and point 1 on positive real axis.
    Returns the remaining 14 points as a flat array of 28 floats.
    """
    p0 = points[0]
    p1 = points[1]
    
    # Translate so p0 is at origin
    pts = points - p0
    p1_new = pts[1]
    
    # Rotate so p1 is on positive real axis
    angle = np.arctan2(p1_new[1], p1_new[0])
    if angle != 0:
        c = np.cos(-angle)
        s = np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
    
    # Ensure p1 is on positive real axis (handle numerical noise)
    pts[1][1] = 0.0
    if pts[1][0] < 0:
        pts[1][0] = -pts[1][0]
        
    # Return flattened array of points 2..15
    return pts[2:].flatten()

def _expand_symmetry(vec):
    """
    Reconstruct 16 points from the 28-dim vector (points 2..15).
    Point 0 is at (0,0), Point 1 is at (r, 0) where r is determined by the vector?
    Wait, the vector only contains points 2..15. We need to recover p1's radius?
    Actually, the relative positions of 2..15 to 0 and 1 are fixed by the vector.
    But p1's distance to p0 is not in the vector.
    Let's check the reduction again.
    If we fix p0=(0,0) and p1=(d,0), the vector contains p2...p15.
    The distance d is a free parameter?
    No, the objective function is scale invariant.
    R = (d_min / d_max)^2.
    If we scale all points by k, d_min and d_max scale by k, so R is invariant.
    Therefore, we can fix the distance between p0 and p1 to be 1.0?
    Yes! Since the score is scale invariant, we can fix p0=(0,0) and p1=(1,0).
    This reduces the degrees of freedom further.
    
    So, _reduce_symmetry should fix p0=(0,0) and p1=(1,0).
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    rest = vec.reshape(14, 2)
    points = np.vstack([p0, p1, rest])
    return points

def _refined_reduce_symmetry(points):
    """
    Fix p0 at (0,0) and p1 at (1,0).
    Returns flat array of p2..p15.
    """
    p0 = points[0]
    p1 = points[1]
    
    # Translate
    pts = points - p0
    p1_new = pts[1]
    
    # Rotate
    angle = np.arctan2(p1_new[1], p1_new[0])
    if angle != 0:
        c = np.cos(-angle)
        s = np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        pts = pts @ rot.T
        
    # Scale so p1 is at (1,0)
    d = np.linalg.norm(pts[1])
    if d < 1e-10:
        # Degenerate case, return random or handle error
        return np.zeros(28)
    pts = pts / d
    
    # Clean up numerical noise for p1
    pts[1] = np.array([1.0, 0.0])
    
    return pts[2:].flatten()

def _nelder_mead_optimize(initial_points, time_budget, rng):
    """
    Run Nelder-Mead optimization on the symmetry-reduced parameterization.
    """
    start_time = time.time()
    
    # Normalize initial points to unit scale for consistency, though NM is scale invariant
    # But we must ensure p0 != p1
    if np.linalg.norm(initial_points[0] - initial_points[1]) < 1e-8:
        # Perturb p1
        initial_points[1] = initial_points[0] + np.array([1e-8, 0])
        
    x0 = _refined_reduce_symmetry(initial_points)
    
    # Objective function for scipy
    def objective(vec):
        points = _expand_symmetry(vec)
        return -_score(points)
        
    # Options for Nelder-Mead
    # maxiter is per run. We will do multiple restarts or a long run.
    # Since we have a time budget, we can't just set maxiter high and hope.
    # We'll use a callback to check time? Or just run a fixed reasonable number of iterations.
    # Nelder-Mead is fast per iteration (no gradients).
    # Let's aim for ~5000 iterations total across the budget.
    
    # Initial simplex size
    # x0 has 28 dims.
    # Default simplex is x0 + 0.05*|x0| + 0.05 for zeros.
    # Let's use a moderate initial step.
    
    # We will run multiple Nelder-Mead runs with slightly perturbed starts if time permits.
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Run 1: Standard
    if time.time() - start_time < time_budget:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
        if time.time() - start_time > time_budget:
            break
            
    # Run 2: Perturbed start
    if time.time() - start_time < time_budget - 10:
        perturbation = rng.standard_normal(28) * 0.05
        x0_pert = x0 + perturbation
        res2 = minimize(objective, x0_pert, method='Nelder-Mead', 
                        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res2.fun < best_f:
            best_f = res2.fun
            best_x = res2.x
            
    # Convert back to points
    best_points = _expand_symmetry(best_x)
    
    # Final check: ensure valid shape
    if best_points.shape != (16, 2):
        return initial_points, -best_f
        
    return best_points, -best_f

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2

    # Hard deadline: 630s wall-clock, leave 30s margin
    deadline = time.time() + 600

    best_points = None
    best_score = -1.0

    rng = np.random.default_rng(42)

    initial_configs = []

    # Config 1: Hexagonal Packing
    initial_configs.append(_generate_hexagonal_16())
    
    # Config 2: Concentric Rings
    initial_configs.append(_generate_concentric_rings_16())
    
    # Config 3: Random
    initial_configs.append(_generate_random_16(rng))
    
    # Config 4: Another Random
    initial_configs.append(_generate_random_16(rng))
    
    # Config 5: Triangular Lattice (approx)
    # Use hexagonal but slightly distorted?
    # Let's just use a grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts_grid = np.column_stack([gx.ravel(), gy.ravel()])
    initial_configs.append(pts_grid)

    # Evaluate and improve each initial config
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break

        # Ensure valid shape
        if init.shape != (n, 2):
            continue

        # Check score of initial
        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        # Local search (Nelder-Mead)
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate time for this config
        # We have 5 configs. Let's give each ~100s if time permits.
        time_budget = min(remaining - 10, 100)
        
        improved_pts, improved_score = _nelder_mead_optimize(init, time_budget, rng)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # If we still have time, do a final polish on the best solution
    if best_points is not None and time.time() < deadline - 20:
        remaining = deadline - time.time()
        if remaining > 20:
            time_budget = min(remaining - 10, 120)
            improved_pts, improved_score = _nelder_mead_optimize(best_points, time_budget, rng)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_pts.copy()

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
