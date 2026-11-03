import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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

def objective_func(flat_points):
    """Objective function for optimization: covering radius in radians"""
    points = flat_points.reshape(N, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    max_radius, _, _ = compute_covering_radius_and_facets(points)
    return max_radius

def lloyd_step(points, Y, n_iters=50):
    """Perform Lloyd's algorithm iterations"""
    for it in range(n_iters):
        # 1. Compute dot products (proxy for distance)
        dot_products = Y @ points.T  # Shape (M, N)
        
        # 2. Assign each Y to nearest point (max dot product)
        nearest_idx = np.argmax(dot_products, axis=1)  # Shape (M,)
        
        # 3. Compute centroids using np.add.at
        sums = np.zeros((N, 3))
        np.add.at(sums, nearest_idx, Y)
        
        # 4. Update points to centroids
        norms = np.linalg.norm(sums, axis=1)
        # Avoid division by zero
        valid = norms > 1e-10
        points[valid] = sums[valid] / norms[valid, np.newaxis]
        
    return points

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    best_points = None
    best_radius = np.pi
    
    # Generate a high-resolution set of test points for Lloyd's algorithm
    # Using a Fibonacci lattice for uniform distribution
    M = 150000
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
    
    # Try multiple restarts with different initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 20.0:  # Reserve 20s for final polish
            break
            
        if restart == 0:
            points = init_points.copy()
        elif restart == 1:
            # Slightly perturbed Fibonacci
            np.random.seed(123)
            points = init_points + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Different Fibonacci offset
            i_alt = np.arange(N) + 0.3
            z_alt = 1.0 - 2.0 * i_alt / N
            phi_alt = np.pi * (1.0 + np.sqrt(5.0)) * i_alt
            r_alt = np.sqrt(1.0 - z_alt * z_alt)
            points = np.column_stack([r_alt * np.cos(phi_alt), r_alt * np.sin(phi_alt), z_alt])
        else:
            # Random perturbations with increasing noise
            np.random.seed(100 + restart)
            noise_level = 0.2 + 0.1 * (restart - 3)
            points = init_points + np.random.randn(N, 3) * noise_level
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Run Lloyd's algorithm
        points = lloyd_step(points, Y, n_iters=80)
        
        # Check current covering radius
        max_radius, radii, simplices = compute_covering_radius_and_facets(points)
        
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
    
    if best_points is None:
        best_points = init_points
    else:
        # Final polish using scipy.optimize.minimize
        if time.time() - start_time < time_limit - 10.0:
            try:
                flat_initial = best_points.flatten()
                result = minimize(
                    objective_func,
                    flat_initial,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                polished_points = result.x.reshape(N, 3)
                norms = np.linalg.norm(polished_points, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                polished_points = polished_points / norms
                
                polished_radius, _, _ = compute_covering_radius_and_facets(polished_points)
                if polished_radius < best_radius:
                    best_radius = polished_radius
                    best_points = polished_points
            except Exception:
                pass
    
    return best_points
