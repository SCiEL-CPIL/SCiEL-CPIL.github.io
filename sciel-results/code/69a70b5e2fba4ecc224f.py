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
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Vectors for the triangle edges
        AB = B - A
        AC = C - A
        
        # Area of triangle: 0.5 * || AB x AC ||
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Handle degenerate facets
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        # Normal vector for each facet
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward (away from origin)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: distance from pole to any vertex (e.g., A)
        # cos(theta) = P @ A
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _energy(pts, grid_pts):
    """
    Compute a proxy energy: sum of squared angular distances to the nearest grid point.
    This is much faster than hull computation and smooth enough for SA.
    """
    # pts: (N, 3)
    # grid_pts: (G, 3)
    
    # Vectorized distance calculation
    # dist_sq[i, j] = || pts[i] - grid_pts[j] ||^2 = 2 - 2 * pts[i] @ grid_pts[j]
    # We want angular distance theta. cos(theta) = pts[i] @ grid_pts[j].
    # Minimizing sum of theta^2 is equivalent to minimizing sum of (arccos(cos_theta))^2.
    # However, arccos is expensive. 
    # A common approximation for small angles is theta^2 ~ 2(1-cos_theta).
    # But for global search, we can just minimize the sum of (1 - cos_theta) which is monotonic with theta for theta in [0, pi].
    # Actually, to be safe and accurate, let's use the chord distance squared: 2 - 2 cos(theta).
    # This is equivalent to minimizing the average squared chord distance.
    
    # Compute dot products
    dots = np.dot(pts, grid_pts.T) # (N, G)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Find min distance (max dot) for each grid point? 
    # No, the energy is usually defined as: for each grid point, find nearest pts.
    # But we have N points and G grid points.
    # The covering radius is max over sphere.
    # A good proxy is: for each of the N points, it "covers" a region.
    # Alternatively, for each grid point, find the nearest of the N points.
    # Let's do: for each grid point g, find min_j (1 - pts[j] @ g).
    # Sum these.
    
    # dots is (N, G). We want for each column (grid point), the max dot product (min distance).
    # max_dots = np.max(dots, axis=0) # (G,)
    # energy = np.sum(1.0 - max_dots)
    
    # This is O(N*G) per evaluation. With G=1000, N=130, this is 130k ops. Very fast.
    
    max_dots = np.max(dots, axis=0)
    energy = np.sum(1.0 - max_dots)
    
    return energy

def _local_polish(pts, deadline):
    """
    Perform a few iterations of facet-based local optimization to refine the SA result.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    
    # Simple gradient descent on the hull facets
    step = 0.05
    for _ in range(50):
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
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            
            curr_r = np.max(ang_radii)
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
            # Find worst facet
            worst_idx = np.argmax(ang_radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    dir_vec = 0.0
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            # Check improvement
            cand_r = _compute_hull_radius(cand)
            if cand_r < curr_r:
                pts = cand
            else:
                step *= 0.9
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Create a dense spherical grid for the energy function
    # Use a Fibonacci spiral with more points for the grid
    G = 2000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial configuration
    pts = _initial_fib()
    current_energy = _energy(pts, grid_pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T0 = 0.5
    Tend = 0.01
    alpha = 0.995 # Cooling rate
    
    # Run SA
    T = T0
    while T > Tend and time.time() < deadline - 10.0:
        # Generate a candidate: move one point randomly
        idx = rng.integers(N)
        # Random direction
        noise = rng.normal(size=3)
        noise /= (np.linalg.norm(noise) + _EPS)
        
        cand = pts.copy()
        # Move point idx in the random direction
        # Step size proportional to T? Or fixed?
        step_size = 0.1 * T / T0 + 0.01
        cand[idx] = cand[idx] + step_size * noise
        cand[idx] /= (np.linalg.norm(cand[idx]) + _EPS)
        
        cand_energy = _energy(cand, grid_pts)
        
        # Acceptance criterion
        delta_e = cand_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            pts = cand
            current_energy = cand_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = cand.copy()
        
        T *= alpha
        
    # Final local polish using the exact hull metric
    if time.time() < deadline - 2.0:
        best_pts, best_r = _local_polish(best_pts, deadline)
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
