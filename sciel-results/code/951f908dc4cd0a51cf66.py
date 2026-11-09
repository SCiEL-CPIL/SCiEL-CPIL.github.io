"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms


def _covering_radius_degrees(points):
    """Compute covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            # Get the facet points
            facet_pts = points[simplex]
            # Find the point on the sphere furthest from this facet
            # The normal to the facet
            if len(simplex) == 3:
                a, b, c = facet_pts
                normal = np.cross(b - a, c - a)
                normal = normal / np.linalg.norm(normal)
                # The center of the spherical cap is in the direction of the normal
                # The covering radius for this facet is the max angular distance
                # from any point on the facet's cap to the nearest point
                # Approximate: the radius is related to the circumradius of the facet
                # on the sphere
                # For a spherical triangle, the covering radius is the max over
                # the incenter/circumcenter regions
                # Simple approach: check the centroid of the facet on the sphere
                centroid = np.mean(facet_pts, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                # The angular distance from centroid to each vertex
                cos_dists = np.clip(np.dot(facet_pts, centroid), -1.0, 1.0)
                max_dist = np.arccos(cos_dists).max()
                max_radius = max(max_radius, max_dist)
        return np.degrees(max_radius)
    except Exception:
        return 90.0  # Fallback: hemisphere


def _objective(points):
    """Objective: covering radius in degrees."""
    return _covering_radius_degrees(points)


def _gradient_ascent_step(points, lr=0.01):
    """Move each point slightly away from its nearest neighbors to reduce covering radius."""
    new_points = points.copy()
    for i in range(N):
        # Compute distances to all other points
        dists = 1.0 - np.sum(points * points[i], axis=1)
        dists[i] = np.inf
        # Find the 3 nearest neighbors
        nearest = np.argsort(dists)[:3]
        # Move away from nearest neighbors
        move = np.zeros(3)
        for j in nearest:
            move -= points[j]
        move = move / (np.linalg.norm(move) + 1e-12)
        new_points[i] = points[i] + lr * move
    return _normalize(new_points)


def _voronoi_optimize(points, max_iters=2000, time_limit=140.0):
    """Iteratively optimize points using gradient-like steps."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _objective(points)
    
    lr = 0.05
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        new_points = _gradient_ascent_step(points, lr)
        new_score = _objective(new_points)
        
        if new_score < best_score:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points
            # Decrease learning rate slowly
            lr = max(0.001, lr * 0.995)
        else:
            # Try smaller step
            lr = max(0.001, lr * 0.9)
    
    return best_points, best_score


def _initial_config():
    """Generate a good initial configuration mixing icosahedral symmetry and spiral."""
    # Start with golden spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add some perturbation to break symmetry
    rng = np.random.RandomState(42)
    noise = rng.randn(N, 3) * 0.05
    pts = pts + noise
    pts = _normalize(pts)
    
    return pts


def construct():
    """Construct 50 points on the unit sphere minimizing covering radius."""
    start_time = time.time()
    
    # Generate initial configuration
    points = _initial_config()
    
    # Run Voronoi-based optimization
    best_points, best_score = _voronoi_optimize(points, max_iters=3000, time_limit=145.0)
    
    # Ensure we return valid points
    if time.time() - start_time > 148.0:
        # Emergency: return current best
        return _normalize(best_points)
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
