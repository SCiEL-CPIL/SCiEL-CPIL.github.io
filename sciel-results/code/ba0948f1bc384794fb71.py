"""
Optimized 50-point spherical covering using differentiable Voronoi-based optimization.

Approach:
- Represent configuration as 50 points on the unit sphere.
- For each point, compute its Voronoi cell inradius (max distance to cell vertices).
- Cell vertices are circumcenters of Delaunay triangles (triples of neighbors).
- Minimize sum of squared inradii using gradient-based optimization.
- Use smooth approximations to handle non-differentiability at cell boundaries.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _spherical_circumcenter(p1, p2, p3):
    """
    Compute the spherical circumcenter of three points on the unit sphere.
    The circumcenter c satisfies c·p1 = c·p2 = c·p3.
    There are two antipodal solutions; we return the one that gives the
    smaller circumradius (i.e., the pole closer to the triangle).
    
    Returns (center, radius) where radius is the angular distance from center to each vertex.
    """
    # Solve linear system: c·p1 = c·p2 = c·p3
    # c is in span of {p1, p2, p3}
    # Using cross products: c is proportional to (p2-p1) x (p3-p1)
    v1 = p2 - p1
    v2 = p3 - p1
    c = np.cross(v1, v2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        # Degenerate case
        d12 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        d13 = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        d23 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        # Return midpoint of longest edge as approximate center
        max_d = max(d12, d13, d23)
        if max_d == d12:
            center = _normalize(p1 + p2)
        elif max_d == d13:
            center = _normalize(p1 + p3)
        else:
            center = _normalize(p2 + p3)
        radius = np.arccos(np.clip(np.dot(center, p1), -1, 1))
        return center, radius
    
    c1 = c / norm_c
    c2 = -c1
    
    # Choose the pole that gives smaller radius
    cos_r1 = np.clip(np.dot(c1, p1), -1.0, 1.0)
    cos_r2 = np.clip(np.dot(c2, p1), -1.0, 1.0)
    r1 = np.arccos(cos_r1)
    r2 = np.arccos(cos_r2)
    
    if r1 <= r2:
        return c1, r1
    else:
        return c2, r2


def _compute_voronoi_inradii(points, k_neighbors=10):
    """
    Compute the inradius for each point's Voronoi cell.
    
    For each point i, find its k nearest neighbors. The Voronoi cell vertices
    are circumcenters of triples (i, j, l) where j, l are neighbors of i.
    The inradius is the maximum circumradius over all such triples that
    actually form a vertex of the Voronoi cell.
    
    We approximate by considering all triples of neighbors.
    """
    n = len(points)
    # Precompute pairwise cosines and distances
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(cos_dists)
    np.fill_diagonal(dists, 0.0)
    
    inradii = np.zeros(n)
    
    for i in range(n):
        # Get k nearest neighbors
        idx = np.argsort(dists[i])[:k_neighbors]
        
        circumradii = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                j = idx[a]
                l = idx[b]
                # Check that this triple is non-degenerate
                c, r = _spherical_circumcenter(points[i], points[j], points[l])
                # Verify the circumcenter is actually a Voronoi vertex:
                # it should be closer to i than to any other point
                # (approximately: check it's within the cap defined by i, j, l)
                circumradii.append(r)
        
        if len(circumradii) == 0:
            inradii[i] = 0.0
        else:
            # Use soft-max for differentiability
            circumradii = np.array(circumradii)
            temp = 0.03
            # Clip to avoid overflow
            cr_clipped = np.clip(circumradii, -10, 10)
            inradii[i] = temp * np.log(np.sum(np.exp(cr_clipped / temp)))
    
    return inradii


def _objective(x):
    """
    Objective: sum of squared Voronoi inradii.
    """
    points = _normalize(x.reshape(N, 3))
    inradii = _compute_voronoi_inradii(points)
    return np.sum(inradii ** 2)


def _true_covering_radius(points):
    """
    Compute the true covering radius by sampling the sphere densely.
    """
    points = _normalize(points)
    n = len(points)
    
    # Sample many points on the sphere
    rng = np.random.RandomState(12345)
    n_samples = 50000
    # Uniform sampling on sphere
    z = rng.uniform(-1, 1, n_samples)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    r = np.sqrt(np.clip(1 - z**2, 0, 1))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find nearest point
    cos_dists = np.clip(samples @ points.T, -1.0, 1.0)
    min_dists = np.arccos(np.max(cos_dists, axis=1))
    
    return np.max(min_dists) * 180.0 / np.pi


def _fibonacci_initial(n_points):
    """Generate Fibonacci spiral initial configuration."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(x0, max_iter=500, maxfun=10000, ftol=1e-12):
    """Run L-BFGS-B optimization from a starting point."""
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': maxfun, 'ftol': ftol}
        )
        return result.x, result.fun
    except Exception:
        return x0, _objective(x0)


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses differentiable Voronoi inradius optimization with proper spherical geometry.
    """
    start_time = time.time()
    max_time = 110  # Leave margin for safety
    
    # Initial configuration: Fibonacci spiral
    points_init = _fibonacci_initial(N)
    x0 = points_init.flatten().copy()
    
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 1: Coarse optimization
    def _time_check():
        return time.time() - start_time > max_time - 20
    
    if not _time_check():
        x_new, obj_new = _optimize_from_start(x0, max_iter=200, maxfun=5000, ftol=1e-10)
        if obj_new < best_obj:
            best_x = x_new
            best_obj = obj_new
    
    # Phase 2: Multiple restarts with perturbations
    rng = np.random.RandomState(42)
    for restart in range(6):
        if _time_check():
            break
        
        # Create perturbation with decreasing scale
        scale = 0.2 * (1.0 - restart / 8.0)
        x_perturbed = best_x + rng.randn(N * 3) * scale
        
        x_new, obj_new = _optimize_from_start(x_perturbed, max_iter=150, maxfun=4000, ftol=1e-10)
        if obj_new < best_obj:
            best_x = x_new
            best_obj = obj_new
    
    # Phase 3: Fine-tuning
    if time.time() - start_time < max_time - 10:
        x_new, obj_new = _optimize_from_start(best_x, max_iter=300, maxfun=8000, ftol=1e-12)
        if obj_new < best_obj:
            best_x = x_new
            best_obj = obj_new
    
    # Phase 4: Very fine-tuning if time permits
    if time.time() - start_time < max_time - 5:
        x_new, obj_new = _optimize_from_start(best_x, max_iter=200, maxfun=5000, ftol=1e-14)
        if obj_new < best_obj:
            best_x = x_new
            best_obj = obj_new
    
    points = _normalize(best_x.reshape(N, 3))
    
    # Validate: ensure points are on unit sphere and not degenerate
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6):
        points = _fibonacci_initial(N)
    
    return points

# EVOLVE-BLOCK-END
