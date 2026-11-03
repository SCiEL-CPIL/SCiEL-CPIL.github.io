"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay


def _covering_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere.
    Uses the circumradius of each facet of the convex hull, but only for
    facets whose outward normal has positive dot product with the facet centroid
    (i.e., facets that are 'outer' and whose cap is empty).
    Actually, for a set of points on the sphere, the covering radius is the
    maximum over all spherical Voronoi cells of the distance from the cell
    center to its farthest vertex. A simpler approach: for each facet of the
    convex hull, compute the circumradius of that facet (the radius of the
    circumscribed circle on the sphere). The covering radius is the maximum
    of these circumradii over all facets.
    
    For a triangular facet with vertices v1, v2, v3 on the unit sphere, the
    circumradius (angular distance from the facet center to a vertex) can be
    computed as follows: the center is the normalized sum of the vertices
    (or more precisely, the point on the sphere closest to the centroid of
    the triangle in 3D). Actually, the circumcenter of a spherical triangle
    is the normalized vector sum of the three vertices if they form an acute
    triangle. For obtuse triangles, the circumcenter may not be well-defined
    in the same way.
    
    A more robust approach: the covering radius is the maximum, over all
    points y on the sphere, of the minimum distance to any configuration point.
    This is equivalent to the maximum circumradius of the spherical Voronoi
    diagram. For the convex hull approach, we can compute for each facet the
    angular distance from the facet's circumcenter to its vertices.
    
    Let's use a simpler but correct method: for each pair of points, and for
    each point, compute the angular distance. But that's O(N^2) which is fine
    for N=130. Actually, the covering radius can be computed by finding the
    point on the sphere farthest from all configuration points. This is a
    non-trivial optimization problem.
    
    Alternative: use the fact that the covering radius is the minimum r such
    that the union of caps of radius r centered at the configuration points
    covers the sphere. We can binary search on r and check coverage.
    
    For efficiency, let's use the convex hull approach. For each facet of the
    convex hull of the points, compute the circumradius of the facet. The
    covering radius is the maximum of these circumradii.
    
    For a spherical triangle with vertices a, b, c (unit vectors), the
    circumradius R satisfies: the distance from the circumcenter to each
    vertex is R. The circumcenter can be found as the normalized vector
    (a + b + c) if the triangle is acute. For a general triangle, we need
    to be more careful.
    
    Let's use a different approach: for each facet, compute the point on the
    sphere that is equidistant from the three vertices. This is the spherical
    circumcenter. For a triangle with vertices a, b, c, the circumcenter is
    in the direction of the cross product (b - a) x (c - a) or similar.
    
    Actually, let's just compute the covering radius by sampling. We'll
    generate a dense set of test points on the sphere and compute the minimum
    distance to the nearest configuration point for each. The maximum of these
    is an upper bound on the covering radius. For the actual evaluation,
    the evaluator will do a proper computation.
    
    For our purposes during optimization, we'll use a proxy objective.
    """
    # Use the convex hull method
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        max_circumradius = 0.0
        for f in facets:
            v1, v2, v3 = pts[f[0]], pts[f[1]], pts[f[2]]
            # Compute the circumradius of the spherical triangle
            # The circumcenter is the normalized sum if the triangle is acute
            center = v1 + v2 + v3
            norm = np.linalg.norm(center)
            if norm < 1e-10:
                # Degenerate case
                center = (v1 + v2 + v3) / 2
                norm = np.linalg.norm(center)
                if norm < 1e-10:
                    continue
                center = center / norm
            else:
                center = center / norm
            
            # Check if the center is on the correct side of the facet
            # The facet normal should point outward
            normal = np.cross(v2 - v1, v3 - v1)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-10:
                continue
            normal = normal / normal_norm
            
            # The center should be on the same side as the facet normal
            # relative to the facet plane
            if np.dot(center, normal) < 0:
                # Use the opposite center (for obtuse triangles)
                center = -center
            
            # Compute the angular distance from center to v1
            cos_angle = np.clip(np.dot(center, v1), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            max_circumradius = max(max_circumradius, angle)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.pi  # Fallback: 180 degrees


def _soft_max_covering_radius(pts, temperature=0.1):
    """Compute a soft approximation of the covering radius using a soft-max
    over pairwise distances. This is differentiable and can be used for
    gradient-based optimization."""
    # Compute all pairwise angular distances
    # For efficiency, we'll compute the distance from each point to all others
    # and take a soft-max over the minimum distance
    
    # Actually, we want to minimize the maximum over y of min_i dist(y, p_i)
    # This is hard to compute directly. Instead, we'll use a proxy:
    # minimize the sum of exp(-dist(p_i, p_j)/temperature) over all pairs,
    # which encourages points to be spread out.
    
    # Better proxy: minimize the maximum pairwise distance (diameter)
    # combined with a term that encourages uniform coverage.
    
    # Let's use a different approach: for each point, compute its distance
    # to its nearest neighbor, and minimize the maximum of these.
    # This is the "packing" objective, which is related to covering.
    
    # Compute pairwise distances
    cosines = np.dot(pts, pts.T)
    np.clip(cosines, -1.0, 1.0, out=cosines)
    angles = np.arccos(cosines)
    
    # For each point, find the minimum distance to any other point
    np.fill_diagonal(angles, np.pi)
    min_distances = np.min(angles, axis=1)
    
    # Soft-max of the minimum distances
    # We want to minimize the maximum, so we use a soft-max
    max_angle = np.max(min_distances)
    soft_max = max_angle + temperature * np.log(np.sum(np.exp((min_distances - max_angle) / temperature)))
    
    return soft_max


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_score = _covering_radius(pts)
    
    # Gradient descent with soft objective
    temperature = 0.1
    lr = 0.01
    
    # Use a simple gradient descent on the sphere
    # The objective is the soft-max covering radius
    # We'll use a finite difference approximation of the gradient
    
    for step in range(5000):
        if time.time() > deadline:
            break
        
        # Compute current score
        current_score = _soft_max_covering_radius(pts, temperature)
        
        # Compute gradient using finite differences
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(3):
                pts_pert = pts.copy()
                pts_pert[i, j] += 1e-4
                # Renormalize
                pts_pert[i] = pts_pert[i] / np.linalg.norm(pts_pert[i])
                score_pert = _soft_max_covering_radius(pts_pert, temperature)
                grad[i, j] = (score_pert - current_score) / 1e-4
        
        # Project gradient onto the tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Update
        pts = pts - lr * grad
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Decay learning rate and temperature
        lr *= 0.999
        temperature *= 0.9995
        
        # Every 100 steps, evaluate the true covering radius
        if step % 100 == 0:
            true_score = _covering_radius(pts)
            if true_score < best_score:
                best_score = true_score
                best_pts = pts.copy()
    
    # Final evaluation
    final_score = _covering_radius(best_pts)
    if final_score < best_score:
        best_score = final_score
    
    return best_pts
# EVOLVE-BLOCK-END
