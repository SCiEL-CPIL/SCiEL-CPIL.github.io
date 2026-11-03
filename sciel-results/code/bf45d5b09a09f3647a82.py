"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START

def _compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        # Degenerate facets, handle gracefully
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices


def _covering_radius(points):
    """Compute the covering radius of a set of points on the unit sphere."""
    points = np.array(points)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    radii, centers, simplices = _compute_facet_data(points)
    
    if len(radii) == 0:
        # Fallback: grid search
        max_dist = 0
        for i in range(80):
            for j in range(80):
                theta = np.pi * i / 80
                phi = 2 * np.pi * j / 80
                y = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
                dists = np.arccos(np.clip(points @ y, -1, 1))
                max_dist = max(max_dist, np.min(dists))
        return np.degrees(max_dist)
    
    return np.degrees(np.max(radii))


def _greedy_init(n_points, seed=0):
    """
    Greedy initialization: start with a random point, then repeatedly add the point
    that maximizes the minimum distance to existing points.
    """
    rng = np.random.default_rng(seed)
    points = [rng.standard_normal(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate many candidates
        candidates = rng.standard_normal((500, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        existing = np.array(points)
        # Dot products
        dots = candidates @ existing.T
        min_cos = np.min(dots, axis=1)
        min_cos = np.clip(min_cos, -1.0, 1.0)
        max_dist = np.arccos(min_cos)
        
        best_idx = np.argmax(max_dist)
        points.append(candidates[best_idx])
    
    return np.array(points)


def _lloyd_spherical(points, n_iter=10, n_neighbors=10):
    """
    Spherical Lloyd's algorithm: move each point towards the centroid of its Voronoi cell.
    Approximated by moving towards the centroid of nearest neighbors.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    for _ in range(n_iter):
        new_points = np.zeros_like(points)
        for i in range(N):
            # Find nearest neighbors
            dists = np.array([np.linalg.norm(points[i] - points[j]) for j in range(N)])
            dists[i] = np.inf
            neighbors = np.argsort(dists)[:n_neighbors]
            
            # Weighted centroid
            weights = 1.0 / (dists[neighbors] + 1e-6)
            centroid = np.sum(points[neighbors] * weights[:, None], axis=0)
            norm_c = np.linalg.norm(centroid)
            if norm_c > 1e-6:
                new_points[i] = centroid / norm_c
            else:
                new_points[i] = points[i]
        
        points = new_points
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    return points


def _facet_driven_refine(points, time_budget):
    """
    Refine points by moving vertices of large facets towards the facet circumcenter.
    This is a direct attack on the covering radius.
    """
    start = time.time()
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    best_local = points.copy()
    best_local_radius = np.pi
    
    alpha = 0.05
    no_improve = 0
    
    while time.time() - start < time_budget:
        radii, centers, simplices = _compute_facet_data(points)
        
        if len(radii) == 0:
            break
            
        max_radius = np.max(radii)
        
        if max_radius < best_local_radius:
            best_local_radius = max_radius
            best_local = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                alpha *= 0.9
                if alpha < 1e-4:
                    break
            if no_improve > 200:
                break
        
        # Identify facets to move (top 20% by radius)
        if len(radii) > 5:
            threshold = np.percentile(radii, 80)
            mask = radii > threshold
        else:
            max_idx = np.argmax(radii)
            mask = np.zeros(len(radii), dtype=bool)
            mask[max_idx] = True
        
        if not np.any(mask):
            break
        
        updates = np.zeros_like(points)
        
        for m in np.where(mask)[0]:
            simplex = simplices[m]
            i_idx, j_idx, k_idx = simplex
            
            a = points[i_idx]
            b = points[j_idx]
            c = points[k_idx]
            
            center = centers[m]
            
            step_size = alpha * radii[m]
            
            updates[i_idx] += step_size * (center - a)
            updates[j_idx] += step_size * (center - b)
            updates[k_idx] += step_size * (center - c)
        
        points += updates
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        alpha *= 0.999
    
    return best_local


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_points = None
    best_radius = np.inf
    
    # Generate multiple initial configurations
    initializations = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib)
    
    # 2. Greedy initialization
    initializations.append(_greedy_init(N, seed=42))
    initializations.append(_greedy_init(N, seed=123))
    
    # 3. Random + Fibonacci noise
    rng = np.random.default_rng(42)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        pts = fib + rng.standard_normal((N, 3)) * 0.2
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        initializations.append(pts)
    
    # 4. Pure random
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    initializations.append(pts)
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        
        # First, run Lloyd's to get a decent starting point
        pts = _lloyd_spherical(init, n_iter=5, n_neighbors=10)
        radius = _covering_radius(pts)
        
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()
        
        # Then, facet-driven refinement
        remaining_time = deadline - time.time() - 15
        if remaining_time > 5:
            pts_refined = _facet_driven_refine(pts, time_budget=min(remaining_time, 25))
            radius_refined = _covering_radius(pts_refined)
            
            if radius_refined < best_radius:
                best_radius = radius_refined
                best_points = pts_refined.copy()
    
    # Final polish with a longer facet-driven refinement if time allows
    if best_points is not None and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 5:
            pts_polish = _facet_driven_refine(best_points, time_budget=remaining_time)
            radius_polish = _covering_radius(pts_polish)
            
            if radius_polish < best_radius:
                best_radius = radius_polish
                best_points = pts_polish.copy()
    
    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
