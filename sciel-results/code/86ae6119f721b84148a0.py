import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Start with a good deterministic structure
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 0.001
    alpha = 0.995 # Cooling rate
    
    current_pts = best_pts.copy()
    current_r = best_r
    T = T_start
    
    # We will iterate until time runs out
    # Each step involves a random perturbation and evaluation
    
    # To make it efficient, we can batch evaluations or just run tight loop
    # ConvexHull is the bottleneck. ~1ms per call? 
    # Let's estimate: 120s / 1ms = 120,000 iterations. Plenty.
    
    while time.time() < deadline:
        # Generate a candidate
        cand = current_pts.copy()
        
        # Randomly select a point to perturb
        idx = rng.integers(0, N)
        
        # Perturbation: move point slightly in random direction on sphere
        # Step size decreases with temperature
        step_size = 0.05 * T / T_start + 0.001
        
        # Random direction
        rand_vec = rng.standard_normal(3)
        rand_vec /= np.linalg.norm(rand_vec)
        
        new_pt = cand[idx] + step_size * rand_vec
        new_pt /= np.linalg.norm(new_pt)
        cand[idx] = new_pt
        
        # Evaluate
        cand_r = _compute_hull_radius(cand)
        
        # Acceptance criterion
        if cand_r < current_r:
            current_pts = cand
            current_r = cand_r
            if current_r < best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(-(delta_r) / T)
            delta_r = cand_r - current_r
            # Avoid overflow/underflow
            if delta_r / T < 700:
                prob = np.exp(-delta_r / T)
            else:
                prob = 0.0
                
            if rng.random() < prob:
                current_pts = cand
                current_r = cand_r
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end

    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
