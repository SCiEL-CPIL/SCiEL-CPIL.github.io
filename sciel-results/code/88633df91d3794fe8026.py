import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    Returns:
        max_radius: float
        radii: (M,) array of circumradii for each facet
        simplices: (M, 3) array of vertex indices for each facet
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # Fallback if hull fails
        return np.pi, np.array([]), np.array([])
    
    radii = np.zeros(len(simplices))
    
    for idx, simplex in enumerate(simplices):
        i, j, k = simplex
        a = points[i]
        b = points[j]
        c = points[k]
        
        # Normal vector to the plane of the triangle
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[idx] = np.pi / 2
            continue
            
        u = n / norm_n
        
        # Ensure u points towards the centroid (inside the spherical triangle)
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
            
        # Circumradius is the angle between u and any vertex
        cos_theta = np.dot(u, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[idx] = np.arccos(cos_theta)
        
    max_radius = np.max(radii) if len(radii) > 0 else np.pi
    return max_radius, radii, simplices

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    best_points = None
    best_radius = np.pi
    
    # Generate a high-resolution set of test points for force calculation
    # Using a Fibonacci lattice for uniform distribution
    M = 200000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial points: Fibonacci sphere
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    init_points = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    
    # Parameters for Molecular Dynamics
    repulsion_strength = 10.0
    attraction_strength = 5.0
    damping = 0.9
    dt = 0.05
    n_steps = 2000
    
    # Try multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
        
        if restart == 0:
            points = init_points.copy()
        else:
            np.random.seed(42 + restart)
            points = init_points + np.random.randn(N, 3) * 0.5
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        velocities = np.zeros((N, 3))
        
        for step in range(n_steps):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            forces = np.zeros((N, 3))
            
            # 1. Repulsive forces between points
            # Compute pairwise distances
            # dist^2 = 2 - 2 * dot
            dot = points @ points.T
            dist_sq = 2.0 - 2.0 * dot
            np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero, will be masked
            
            # Force is proportional to 1 / dist^2 * (direction)
            # Direction is (x_j - x_i)
            # We want to push i away from j if they are close.
            # F_i = sum_j (x_i - x_j) / |x_i - x_j|^2
            # This is equivalent to gradient of -1/d
            
            # Efficient calculation of repulsion
            # We only care about close pairs to save time? Or just do full O(N^2) since N=130 is small.
            # N^2 = 16900 operations, very fast.
            
            # dist_sq is (N, N)
            # We want F_i = sum_j (x_i - x_j) * w_ij
            # where w_ij = 1 / dist_sq_ij^2 (or similar)
            
            # Let's use w_ij = 1 / dist_sq_ij
            # F_i = sum_j (x_i - x_j) / dist_sq_ij
            # Note: (x_i - x_j) is a vector.
            
            # We can compute this using matrix operations.
            # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :] # (N, N, 3)
            # This is memory heavy (130*130*3 ~ 50k floats, fine).
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :] # (N, N, 3)
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            dist_sq_masked = np.where(mask, dist_sq, 1.0)
            
            # Weight matrix
            # Use a softer potential to avoid instability
            # w = 1 / (dist_sq + eps)^2
            eps = 1e-6
            w = repulsion_strength / (dist_sq_masked + eps)**2
            np.fill_diagonal(w, 0.0)
            
            # F_i = sum_j w_ij * (x_i - x_j)
            # forces = np.sum(w[:, :, np.newaxis] * diff, axis=1)
            forces += np.sum(w[:, :, np.newaxis] * diff, axis=1)
            
            # 2. Attractive forces to grid points (Y)
            # For each point x_i, find nearest grid point y_k.
            # Force = attraction_strength * (y_k - x_i)
            # This pulls points to fill holes.
            
            dot_grid = points @ Y.T # (N, M)
            nearest_idx = np.argmax(dot_grid, axis=1) # (N,)
            nearest_Y = Y[nearest_idx] # (N, 3)
            
            # Force is towards nearest_Y
            forces += attraction_strength * (nearest_Y - points)
            
            # Update velocities and positions (Velocity Verlet simplified)
            velocities = velocities * damping + forces * dt
            points = points + velocities * dt
            
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            points = points / norms
            
            # Project velocity to tangent plane?
            # v_new = v - (v . p) p
            dot_vp = np.sum(velocities * points, axis=1, keepdims=True)
            velocities = velocities - dot_vp * points
            
            # Check current covering radius periodically
            if step % 100 == 0:
                max_radius, _, _ = compute_covering_radius_and_facets(points)
                if max_radius < best_radius:
                    best_radius = max_radius
                    best_points = points.copy()
    
    if best_points is None:
        best_points = init_points
    
    return best_points
