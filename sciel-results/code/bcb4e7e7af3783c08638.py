"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _compute_covering_radius_and_cells(points):
    """
    Compute covering radius and Voronoi cell centroids using convex hull.
    Returns: (max_radius, cell_centroids) where cell_centroids[i] is the centroid of point i's Voronoi cell.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, None
    
    max_radius = 0.0
    n_facets = len(hull.simplices)
    
    # For each point, collect circumcenters of facets that include it
    # Initialize centroids as zeros
    centroid_sums = np.zeros((N, 3))
    centroid_counts = np.zeros(N)
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute circumcenter of triangle on sphere
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        
        c_circ = n / norm_n
        
        # Ensure correct sign: circumcenter should be on same side as centroid of triangle
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c_circ, centroid) < 0:
            c_circ = -c_circ
        
        # Compute radius (angular distance from circumcenter to any vertex)
        cos_theta = np.clip(np.dot(c_circ, v1), -1.0, 1.0)
        radius = np.arccos(cos_theta)
        
        if radius > max_radius:
            max_radius = radius
        
        # Accumulate circumcenter into each vertex's Voronoi cell centroid
        for idx in simplex:
            centroid_sums[idx] += c_circ
            centroid_counts[idx] += 1
    
    # Compute normalized centroids
    cell_centroids = np.zeros((N, 3))
    for i in range(N):
        if centroid_counts[i] > 0:
            cell_centroids[i] = centroid_sums[i] / centroid_counts[i]
            norm = np.linalg.norm(cell_centroids[i])
            if norm > 1e-10:
                cell_centroids[i] /= norm
            else:
                cell_centroids[i] = points[i]
        else:
            cell_centroids[i] = points[i]
    
    return max_radius, cell_centroids

def _grid_covering_radius(points):
    """Fallback: estimate covering radius by sampling the sphere."""
    n_theta = 50
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    
    test_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    diff = test_points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_dists = np.min(dists, axis=1)
    
    cos_angles = 1.0 - 0.5 * min_dists ** 2
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)
    
    return np.max(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    
    # Compute initial covering radius
    best_radius, _ = _compute_covering_radius_and_cells(points)
    
    # Phase 1: Lloyd's algorithm - move each point toward its Voronoi cell centroid
    # Phase 2: Fine-tuning with small random perturbations around the best found
    
    np.random.seed(42)
    
    # Lloyd's algorithm parameters
    lloyd_steps = 500
    lloyd_alpha_start = 0.3
    lloyd_alpha_end = 0.01
    lloyd_decay = (lloyd_alpha_end / lloyd_alpha_start) ** (1.0 / max(lloyd_steps - 1, 1))
    
    current_alpha = lloyd_alpha_start
    
    for step in range(lloyd_steps):
        if time.time() > deadline - 20.0:  # Leave 20s for phase 2
            break
        
        radius, cell_centroids = _compute_covering_radius_and_cells(points)
        
        if cell_centroids is None:
            # Hull failed, use grid fallback
            radius = _grid_covering_radius(points)
            break
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Move each point toward its Voronoi cell centroid
        for i in range(N):
            direction = cell_centroids[i] - points[i]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction /= norm_dir
                new_point = points[i] + current_alpha * direction
                new_norm = np.linalg.norm(new_point)
                if new_norm > 1e-10:
                    points[i] = new_point / new_norm
        
        current_alpha *= lloyd_decay
    
    # Phase 2: Fine-tuning with small random perturbations
    remaining_time = deadline - 20.0 - time.time()
    if remaining_time > 5.0:
        # Reset to best found
        points = best_points.copy()
        best_radius, _ = _compute_covering_radius_and_cells(points)
        
        fine_tune_steps = 2000
        fine_tune_alpha = 0.005
        
        for step in range(fine_tune_steps):
            if time.time() > deadline - 2.0:
                break
            
            # Small random perturbation
            perturbations = np.random.randn(N, 3) * fine_tune_alpha
            new_points = points + perturbations
            norms = np.linalg.norm(new_points, axis=1)
            valid = norms > 1e-10
            if not np.all(valid):
                continue
            new_points = new_points / norms[:, np.newaxis]
            
            new_radius, _ = _compute_covering_radius_and_cells(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points.copy()
    
    # Final validation
    final_radius, _ = _compute_covering_radius_and_cells(best_points)
    if final_radius > np.pi / 2:  # Check for hemisphere issue
        # Try a different initial configuration
        np.random.seed(123)
        alt_points = _fibonacci_sphere(N) + np.random.randn(N, 3) * 0.1
        alt_norms = np.linalg.norm(alt_points, axis=1)
        alt_points = alt_points / alt_norms[:, np.newaxis]
        alt_radius, _ = _compute_covering_radius_and_cells(alt_points)
        if alt_radius < final_radius:
            best_points = alt_points
            best_radius = alt_radius
    
    return best_points

# EVOLVE-BLOCK-END
