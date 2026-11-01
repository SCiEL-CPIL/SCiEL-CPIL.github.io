"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time

def _covering_radius(points):
    """Compute covering radius via convex hull facets.
    For each facet (triangle), check if the cap centered at the facet's circumcenter
    (projected to sphere) is empty. The covering radius is the max over all facets
    of the angular distance from the facet's circumcenter to its vertices."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_angle = 0.0
    for simplex in hull.simplices:
        # Get the three vertices of this facet
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the circumcenter of the triangle on the sphere
        # The circumcenter of a spherical triangle is in the direction of
        # v1 + v2 + v3 (for small triangles) or more precisely the cross product approach
        # Actually, for a triangle on a sphere, the circumcenter is along the normal
        # to the plane containing the three points, projected to the sphere.
        
        # Normal to the plane of the triangle
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        # Check if this normal points outward (away from origin)
        # The centroid of the triangle should be on the same side
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        # The circumcenter on the sphere is in the direction of `normal`
        # (for points on the unit sphere, the plane's normal from origin gives the pole)
        
        # Angular distance from circumcenter to each vertex
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        
        # The covering radius contribution from this facet is the max angular distance
        # But we need to verify the cap is empty (no other point is closer to the circumcenter)
        # For now, use the max of the three distances as the facet's circumradius
        facet_radius = max(d1, d2, d3)
        max_angle = max(max_angle, facet_radius)
    
    return max_angle

def _fibonacci_sphere(n, offset=0.5):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _repulsion_optimize(points, max_iters=500, lr=0.01, deadline=None):
    """Optimize points on sphere using repulsion from nearest neighbors.
    Each point is pushed away from its nearest neighbor to reduce local clustering."""
    points = points.copy().astype(float)
    n = len(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        # For each point, find its nearest neighbor and compute repulsive force
        # Force: push point i away from its nearest neighbor j
        # Use soft repulsion: for each pair within some distance, repel
        
        # Efficient: for each point, compute distance to all others
        # d[i,j] = angle between points[i] and points[j]
        # Use dot products
        dots = np.clip(points @ points.T, -1, 1)
        angles = np.arccos(dots)
        
        # Set diagonal to 0
        np.fill_diagonal(angles, 0)
        
        # For each point, the repulsive force is sum of vectors away from nearby points
        # weighted by some kernel
        # Use: force_i = sum_j (points[i] - points[j]) / (1 + angles[i,j]^2) for angles < threshold
        threshold = np.pi / 3  # only repel within 60 degrees
        mask = angles < threshold
        np.fill_diagonal(mask, False)
        
        # Compute forces
        # force_i = sum_j w_ij * (p_i - p_j) where w_ij = 1/(1 + angle_ij^2) if angle < thresh
        weights = np.where(mask, 1.0 / (1.0 + angles**2), 0.0)
        
        # force_i = sum_j weights[i,j] * p_i - sum_j weights[i,j] * p_j
        #        = (sum_j weights[i,j]) * p_i - (weights @ points)[i]
        weight_sums = weights.sum(axis=1, keepdims=True)
        forces = weight_sums * points - (weights @ points)
        
        # Normalize forces and apply
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms > 1e-10, force_norms, 1.0)
        forces = forces / force_norms
        
        # Project onto tangent plane: remove radial component
        # tangent_force = force - (force . p) * p
        radial = (forces * points).sum(axis=1, keepdims=True) * points
        tangent_forces = forces - radial
        
        # Update
        points = points + lr * tangent_forces
        
        # Re-normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Multi-start: try different initializations and pick the best
    best_points = None
    best_radius = np.pi
    
    # Start 1: Fibonacci spiral
    points = _fibonacci_sphere(N)
    radius = _covering_radius(points)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Optimize from this start
    if time.time() < deadline - 30:
        opt_points = _repulsion_optimize(points, max_iters=300, lr=0.005, deadline=deadline - 25)
        opt_radius = _covering_radius(opt_points)
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_points = opt_points.copy()
    
    # Start 2: Different Fibonacci offset
    if time.time() < deadline - 30:
        points2 = _fibonacci_sphere(N, offset=0.1)
        radius2 = _covering_radius(points2)
        if radius2 < best_radius:
            best_radius = radius2
            best_points = points2.copy()
        
        if time.time() < deadline - 25:
            opt_points2 = _repulsion_optimize(points2, max_iters=300, lr=0.005, deadline=deadline - 20)
            opt_radius2 = _covering_radius(opt_points2)
            if opt_radius2 < best_radius:
                best_radius = opt_radius2
                best_points = opt_points2.copy()
    
    # Start 3: Random initialization with optimization
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            # Random points on sphere
            rand_points = rng.normal(size=(N, 3))
            rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
            
            rand_radius = _covering_radius(rand_points)
            
            # Optimize
            if time.time() < deadline - 15:
                opt_rand = _repulsion_optimize(rand_points, max_iters=400, lr=0.008, deadline=deadline - 10)
                opt_rand_radius = _covering_radius(opt_rand)
                if opt_rand_radius < best_radius:
                    best_radius = opt_rand_radius
                    best_points = opt_rand.copy()
    
    # Final refinement on best solution
    if best_points is not None and time.time() < deadline - 5:
        final = _repulsion_optimize(best_points, max_iters=200, lr=0.002, deadline=deadline - 2)
        final_radius = _covering_radius(final)
        if final_radius < best_radius:
            best_points = final
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
