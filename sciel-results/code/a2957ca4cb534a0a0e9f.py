"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _fibonacci_sphere(n):
    """Generate n points on the sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _circumcenter_sphere(p1, p2, p3):
    """
    Calculate the circumcenter of a spherical triangle defined by three points.
    The circumcenter is the point on the sphere equidistant to p1, p2, p3.
    """
    # The circumcenter is proportional to p1 + p2 + p3 if the triangle is small? No.
    # The circumcenter lies in the plane defined by p1, p2, p3.
    # It is also perpendicular to the cross products (p1 x p2) and (p2 x p3).
    # Actually, the circumcenter v satisfies v.p1 = v.p2 = v.p3.
    # This implies v is in the null space of the system:
    # v . (p1 - p2) = 0
    # v . (p1 - p3) = 0
    # v . v = 1
    
    # A simpler geometric approach for unit sphere points:
    # The circumcenter is the normalized sum of the vertices only if they form an equilateral triangle? No.
    # The vector from the origin to the circumcenter is parallel to the cross product of the edges?
    # Let's use the formula:
    # v = normalize( (|p2|^2 (p1 x p3) + |p3|^2 (p1 x p2) + |p1|^2 (p2 x p3)) / 2 ) ... wait, this is for 2D planar.
    
    # On the sphere, the circumcenter is the intersection of the sphere and the plane perpendicular to the chord midpoints?
    # Actually, the circumcenter of a spherical triangle is the pole of the great circle passing through the midpoints of the sides? No.
    
    # Let's solve the linear system:
    # v . p1 = d
    # v . p2 = d
    # v . p3 = d
    # v . v = 1
    # Subtracting: v . (p1 - p2) = 0 and v . (p1 - p3) = 0.
    # So v is orthogonal to (p1-p2) and (p1-p3).
    # Therefore, v is parallel to (p1-p2) x (p1-p3).
    # Let n = (p1-p2) x (p1-p3).
    # v = +/- n / ||n||.
    # We need to determine the sign. The circumcenter should be "inside" the triangle in some sense, or at least on the same side of the great circle as the triangle's interior.
    # Actually, for a covering problem, we want the point furthest from the vertices.
    # The circumcenter is unique up to sign. One is the "center" of the hole, the other is the antipode.
    # We should pick the one that is NOT the antipode of the centroid?
    # Let's check the dot product with the centroid c = (p1+p2+p3)/3.
    # If v . c > 0, v is the "inner" circumcenter.
    
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    p3 = p3 / np.linalg.norm(p3)
    
    v1 = p1 - p2
    v2 = p1 - p3
    n = np.cross(v1, v2)
    
    if np.linalg.norm(n) < 1e-12:
        # Degenerate triangle, return centroid
        return (p1 + p2 + p3) / np.linalg.norm(p1 + p2 + p3)
        
    n = n / np.linalg.norm(n)
    centroid = (p1 + p2 + p3) / 3.0
    if np.dot(n, centroid) < 0:
        n = -n
        
    return n

def _compute_circumcenters(points, hull):
    """
    Compute circumcenters and circumradii for all facets of the convex hull.
    Returns:
    centers: array of shape (M, 3)
    radii: array of shape (M,)
    facets: array of shape (M, 3) containing indices
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for i in range(M):
        idx = facets[i]
        p1 = points[idx[0]]
        p2 = points[idx[1]]
        p3 = points[idx[2]]
        
        c = _circumcenter_sphere(p1, p2, p3)
        centers[i] = c
        
        # The angular distance is the arccos of the dot product
        # We want the max distance to the nearest point among p1, p2, p3
        # Since c is the circumcenter, dist(c, p1) == dist(c, p2) == dist(c, p3)
        # So we just compute dist(c, p1)
        dot = np.dot(c, p1)
        dot = np.clip(dot, -1.0, 1.0)
        radii[i] = np.arccos(dot)
        
    return centers, radii, facets

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Optimization loop
    step_size = 0.1
    min_step_size = 0.001
    
    while time.time() < deadline:
        # Compute convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails (e.g., points are not in general position or all in hemisphere),
            # fall back to a simple repulsion step
            # Simple repulsion: move each point away from its nearest neighbors
            for i in range(N):
                dists = np.linalg.norm(points - points[i], axis=1)
                dists[i] = np.inf
                # Find 5 nearest neighbors
                nn_idx = np.argsort(dists)[:5]
                force = np.zeros(3)
                for j in nn_idx:
                    d = points[i] - points[j]
                    force += d / np.linalg.norm(d)
                points[i] += step_size * force
                points[i] /= np.linalg.norm(points[i])
            continue
            
        centers, radii, facets = _compute_circumcenters(points, hull)
        
        # Identify the largest facets
        # We want to reduce the maximum radius.
        # Let's find the facets with the largest radii.
        max_radius = np.max(radii)
        
        # If max radius is very small, we can break or reduce step size
        if max_radius < 0.001:
            break
            
        # Determine which points to move and how.
        # For each facet, if its radius is close to the max, move its vertices towards the circumcenter.
        # We accumulate a force vector for each point.
        forces = np.zeros((N, 3))
        
        # Threshold: consider facets with radius > 0.9 * max_radius
        threshold = 0.9 * max_radius
        mask = radii > threshold
        
        for i in range(len(facets)):
            if not mask[i]:
                continue
                
            c = centers[i]
            idx = facets[i]
            p1 = points[idx[0]]
            p2 = points[idx[1]]
            p3 = points[idx[2]]
            
            # Move each vertex towards the circumcenter
            # Weight by how much "larger" this facet is compared to average?
            # Or just a fixed weight.
            w = 1.0
            
            # Vector from p1 to c
            v1 = c - p1
            v2 = c - p2
            v3 = c - p3
            
            forces[idx[0]] += w * v1
            forces[idx[1]] += w * v2
            forces[idx[2]] += w * v3
            
        # Apply forces
        # Normalize forces to prevent huge jumps
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        normalized_forces = forces / force_norms
        
        points = points + step_size * normalized_forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Adaptive step size: if improvement is stalled, reduce step size
        # We don't track improvement easily here, so we just decay step size slowly
        step_size *= 0.99
        
        if step_size < min_step_size:
            break

    return points

# EVOLVE-BLOCK-END
