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
    Compute the true covering radius using the convex hull.
    This is the ground truth for scoring.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter degenerate
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The pole is the normalized normal.
        # Distance is arccos(pole . vertex)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        return np.max(radii)
    except Exception:
        return np.pi

def _lloyd_relax(pts, grid, n_iters=50):
    """
    Lloyd's Algorithm for spherical covering.
    1. Compute Voronoi cells (approximated by nearest neighbor on grid).
    2. Move each point to the centroid of its cell.
    """
    pts = pts.copy()
    M = grid.shape[0]
    
    for _ in range(n_iters):
        # Compute nearest point for each grid point
        # dots: (M, N)
        dots = grid @ pts.T
        # Nearest point corresponds to max dot product
        nearest_idx = np.argmax(dots, axis=1)
        
        # Compute centroids using bincount for vectorization
        # We need sum of grid points for each bin
        # new_pts[i] = sum(grid[nearest_idx == i])
        # Using np.bincount with weights is tricky for vectors, so we do:
        # new_pts = np.zeros((N, 3))
        # for k in range(3):
        #     new_pts[:, k] = np.bincount(nearest_idx, weights=grid[:, k], minlength=N)
        
        new_pts = np.zeros((N, 3))
        for k in range(3):
            new_pts[:, k] = np.bincount(nearest_idx, weights=grid[:, k], minlength=N)
            
        counts = np.bincount(nearest_idx, minlength=N).astype(float)
        
        # Normalize centroids
        valid = counts > 0
        norms = np.linalg.norm(new_pts[valid], axis=1, keepdims=True)
        new_pts[valid] /= (norms + _EPS)
        
        # If a point has no assigned grid points, keep it
        new_pts[~valid] = pts[~valid]
        
        # Check for convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < 1e-8:
            break
            
    return pts

def _generate_grid(M):
    """Generate a Fibonacci lattice grid."""
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Multi-restart strategy with increasing grid resolution
    # The key structural change: instead of a fixed grid, we use a hierarchy of grids
    # and a more aggressive multi-restart with random initializations, focusing on
    # the fact that Lloyd's algorithm is very effective for this problem.
    
    # Pre-compute grids of different resolutions
    grid_coarse = _generate_grid(5000)
    grid_medium = _generate_grid(20000)
    grid_fine = _generate_grid(50000)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    n_restarts = 20
    
    # Time allocation:
    # - 20 restarts
    # - Each restart: 500 iters on coarse, 50 iters on medium, 20 iters on fine
    # - Plus hull checks
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
            
        # Phase 1: Coarse relaxation
        # Use a time budget for this phase
        phase_deadline = time.time() + 2.0
        iters = 0
        while time.time() < phase_deadline and iters < 200:
            pts = _lloyd_relax(pts, grid_coarse, n_iters=10)
            iters += 10
            
        # Phase 2: Medium relaxation
        phase_deadline = time.time() + 4.0
        iters = 0
        while time.time() < phase_deadline and iters < 100:
            pts = _lloyd_relax(pts, grid_medium, n_iters=5)
            iters += 5
            
        # Phase 3: Fine relaxation
        phase_deadline = time.time() + 3.0
        iters = 0
        while time.time() < phase_deadline and iters < 50:
            pts = _lloyd_relax(pts, grid_fine, n_iters=2)
            iters += 2
            
        # Evaluate
        curr_r = _compute_hull_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
    # Final polish on the best configuration with a very fine grid
    if time.time() < deadline - 5.0:
        grid_very_fine = _generate_grid(100000)
        phase_deadline = time.time() + 10.0
        while time.time() < phase_deadline:
            best_pts = _lloyd_relax(best_pts, grid_very_fine, n_iters=5)
            curr_r = _compute_hull_radius(best_pts)
            if curr_r < best_r:
                best_r = curr_r
            else:
                # If no improvement, break to save time
                break
                
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
