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

def _compute_hull_radius(pts):
    """
    Compute the covering radius (in radians) by finding the maximum circumradius
    of the convex hull facets.
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
        
        # Side lengths
        len_AB = np.linalg.norm(AB, axis=1)
        len_AC = np.linalg.norm(AC, axis=1)
        len_BC = np.linalg.norm(C - B, axis=1)
        
        # Area
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Circumradius R = abc / (4A)
        prod_sides = len_AB * len_AC * len_BC
        denom = 4.0 * area
        
        valid = denom > _EPS
        radii = np.full(M, np.inf)
        radii[valid] = prod_sides[valid] / denom[valid]
        
        # Normals
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        centroids = (A + B + C) / 3.0
        signs = np.sum(cross * centroids, axis=1, keepdims=True)
        normals = cross * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular radius
        cos_theta = np.sum(normals * A, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        ang_radii[~valid] = np.inf
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _energy(pts, grid):
    """
    Global energy function: sum of squared angular distances from grid points to nearest point in pts.
    This encourages uniform coverage and avoids local minima of the hull-based metric.
    """
    # grid: (G, 3) unit vectors
    # pts: (N, 3) unit vectors
    
    # Compute all pairwise dot products: (G, N)
    # cos(theta_ij) = grid[i] @ pts[j]
    dots = grid @ pts.T
    
    # Clip for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each grid point i, find the nearest point j (max dot product)
    max_dots = np.max(dots, axis=1)
    
    # Angular distance squared: (arccos(max_dot))^2
    # We can approximate or use arccos directly. arccos is expensive but G is moderate.
    # To speed up, we can use a proxy: (1 - max_dot)^2 or similar.
    # Let's use (1 - max_dot) as a proxy for distance, squared.
    # This is monotonic with angle for small angles and avoids arccos.
    # dist_proxy = 1 - max_dots
    # energy = sum(dist_proxy^2)
    
    dists = 1.0 - max_dots
    return np.sum(dists * dists)

def _local_polish(pts, deadline, rng):
    """
    Local refinement using the facet-pole method (similar to current but faster/simpler).
    """
    pts = np.array(pts, dtype=float)
    best = pts.copy()
    best_r = _compute_hull_radius(best)
    
    step = 0.05
    for _ in range(200):
        if time.time() > deadline:
            break
            
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
            area = 0.5 * np.linalg.norm(cross, axis=1)
            
            len_AB = np.linalg.norm(AB, axis=1)
            len_AC = np.linalg.norm(AC, axis=1)
            len_BC = np.linalg.norm(C - B, axis=1)
            
            prod_sides = len_AB * len_AC * len_BC
            denom = 4.0 * area
            valid = denom > _EPS
            radii = np.full(M, np.inf)
            radii[valid] = prod_sides[valid] / denom[valid]
            
            norms = np.linalg.norm(cross, axis=1, keepdims=True)
            centroids = (A + B + C) / 3.0
            signs = np.sum(cross * centroids, axis=1, keepdims=True)
            normals = cross * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            ang_radii[~valid] = np.inf
            
            curr_r = np.max(ang_radii)
        except Exception:
            break
            
        if not np.isfinite(curr_r):
            break
            
        if curr_r < best_r:
            best_r = curr_r
            best = pts.copy()
            
        worst_idx = np.argmax(ang_radii)
        if not np.isfinite(ang_radii[worst_idx]):
            break
            
        v_indices = faces[worst_idx]
        pole = normals[worst_idx]
        
        cand = pts.copy()
        for v in v_indices:
            dir_vec = pole - pts[v]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > _EPS:
                dir_vec /= dir_norm
            else:
                continue
            cand[v] = pts[v] + step * dir_vec
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
            
        cand_r = _compute_hull_radius(cand)
        if np.isfinite(cand_r) and cand_r < curr_r:
            pts = cand
        else:
            step *= 0.9
            if step < 1e-4:
                break
                
    return best, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Create a dense spherical grid for the energy function
    # Use Fibonacci grid with ~5000 points
    G = 5000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = _initial_fib()
    best_hull_r = _compute_hull_radius(best_pts)
    
    # Simulated Annealing
    T = 0.05
    T_min = 1e-4
    cooling = 0.999
    
    # Initialize with Fibonacci
    pts = _initial_fib()
    curr_e = _energy(pts, grid)
    
    # Main SA loop
    while time.time() < deadline - 20.0:
        # Propose a move
        idx = rng.integers(N)
        noise = rng.normal(size=3) * 0.1
        cand = pts.copy()
        cand[idx] = cand[idx] + noise
        cand[idx] /= (np.linalg.norm(cand[idx]) + _EPS)
        
        cand_e = _energy(cand, grid)
        
        # Acceptance criterion
        if cand_e < curr_e or rng.random() < np.exp(-(cand_e - curr_e) / T):
            pts = cand
            curr_e = cand_e
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart temperature occasionally to escape local minima
            if rng.random() < 0.1:
                T = 0.02
                
    # Local polish the best found configuration
    polished, polished_r = _local_polish(pts, deadline, rng)
    
    if polished_r < best_hull_r:
        best_pts = polished
        best_hull_r = polished_r
        
    # Also try polishing the initial Fibonacci
    fib_polished, fib_r = _local_polish(_initial_fib(), deadline, rng)
    if fib_r < best_hull_r:
        best_pts = fib_polished
        best_hull_r = fib_r
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
