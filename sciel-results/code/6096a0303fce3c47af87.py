import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius_hull(pts):
    """
    Compute the exact covering radius using the convex hull method.
    This is expensive, so use sparingly (e.g., for final validation or every N steps).
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        centroids = (A + B + C) / 3.0
        signs = np.sum(cross * centroids, axis=1, keepdims=True)
        normals = cross * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _estimate_covering_radius(pts, probe_points):
    """
    Estimate covering radius by checking max distance to nearest neighbor 
    in a dense set of probe points. This is O(N * P) but vectorized.
    """
    # cdist computes pairwise distances. 
    # We want the max over probes of min over points.
    # dists: (P, N)
    dists = cdist(probe_points, pts)
    min_dists = np.min(dists, axis=1)
    return np.max(min_dists)

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    # Create a dense set of probe points to estimate covering radius quickly.
    # 20000 points is a good balance between accuracy and speed.
    # Using a Fibonacci spiral for probes as well for uniform distribution.
    P = 20000
    i = np.arange(P) + 0.5
    z = 1.0 - 2.0 * i / P
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    probe_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Start with Fibonacci
    current_pts = _initial_fib()
    current_r_est = _estimate_covering_radius(current_pts, probe_points)
    
    best_pts = current_pts.copy()
    best_r_est = current_r_est
    best_r_hull = _compute_covering_radius_hull(current_pts)
    
    # Simulated Annealing Parameters
    T = 0.05  # Initial temperature (step size magnitude)
    cooling_rate = 0.9995
    min_T = 0.001
    
    start_time = time.time()
    iter_count = 0
    last_hull_check = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Generate a perturbation
        # We perturb a random subset of points to allow local exploration
        # or all points for global exploration.
        if rng.random() < 0.5:
            # Global small step
            noise = rng.normal(size=(N, 3)) * T
            cand_pts = current_pts + noise
        else:
            # Local step: move 5 random points more aggressively
            cand_pts = current_pts.copy()
            idx = rng.choice(N, size=5, replace=False)
            noise = rng.normal(size=(5, 3)) * (T * 2.0)
            cand_pts[idx] = cand_pts[idx] + noise
        
        # Normalize to sphere
        norms = np.linalg.norm(cand_pts, axis=1, keepdims=True)
        cand_pts = cand_pts / (norms + _EPS)
        
        # Estimate new covering radius
        cand_r_est = _estimate_covering_radius(cand_pts, probe_points)
        
        # Metropolis acceptance criterion
        delta_r = cand_r_est - current_r_est
        if delta_r < 0 or rng.random() < np.exp(-delta_r / T):
            current_pts = cand_pts
            current_r_est = cand_r_est
            
            # Check if we found a new best
            if current_r_est < best_r_est:
                best_r_est = current_r_est
                best_pts = current_pts.copy()
                
                # Periodically verify with exact hull radius
                # Doing this every 1000 steps or when we find a significant improvement
                if iter_count % 1000 == 0 or current_r_est < best_r_hull - 0.001:
                    r_hull = _compute_covering_radius_hull(best_pts)
                    if r_hull < best_r_hull:
                        best_r_hull = r_hull
                        last_hull_check = iter_count
        else:
            # If rejected, we still cool down
            pass
            
        # Cool down
        T = max(T * cooling_rate, min_T)
        
        # If T is very low, we might be stuck. Restart with a higher T?
        # Or just let it run. Given the time limit, simple SA is fine.
        
        # Safety check: if no improvement in hull radius for a long time, 
        # we might consider a different strategy, but let's stick to SA.
        
    # Final verification of the best candidate found
    final_r_hull = _compute_covering_radius_hull(best_pts)
    
    # If the final hull check is worse than the initial fib hull check (shouldn't happen 
    # if we track best_r_hull correctly, but just in case), use the better one.
    # Actually, best_pts is the one that gave best_r_est. 
    # It's possible best_r_est corresponds to a point that has a slightly worse hull radius 
    # due to estimation error. So we should compare best_pts hull radius vs initial fib hull radius.
    
    init_r_hull = _compute_covering_radius_hull(_initial_fib())
    
    if final_r_hull < init_r_hull:
        return best_pts
    else:
        return _initial_fib()
