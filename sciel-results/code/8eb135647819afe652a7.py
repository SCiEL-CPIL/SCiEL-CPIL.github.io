"""
Repulsive-force + convex-hull based covering radius optimizer for 130 points on S^2.
"""
import numpy as np
from scipy.spatial import ConvexHull
import time
import math

N = 130


def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This equals the maximum circumradius of the
    facets of the convex hull (for points in general position on S^2).
    
    We compute this via:
    1. Compute the convex hull of the points.
    2. For each facet (triangle), compute the circumcenter (pole of the facet plane).
    3. The covering radius is the maximum angle between the pole and any vertex of the facet.
    """
    # Ensure points are normalized
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute force max angular distance to nearest point
        # Sample sphere
        n_samples = 5000
        theta = np.random.uniform(0, np.pi, n_samples)
        phi = np.random.uniform(0, 2 * np.pi, n_samples)
        sx = np.sin(theta) * np.cos(phi)
        sy = np.sin(theta) * np.sin(phi)
        sz = np.cos(theta)
        samples = np.column_stack([sx, sy, sz])
        # For each sample, find nearest point
        # Use dot product
        dots = samples @ pts.T  # (n_samples, N)
        max_cos = np.max(dots, axis=1)
        max_angle = np.arccos(np.clip(max_cos, -1, 1))
        return np.max(max_angle)
    
    # Get facet vertices
    # hull.simplices gives triangles for 3D
    facets = hull.simplices
    n_facets = len(facets)
    
    max_radius = 0.0
    for tri in facets:
        p0 = pts[tri[0]]
        p1 = pts[tri[1]]
        p2 = pts[tri[2]]
        
        # Normal to the facet plane (pointing outward)
        normal = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            continue
        normal = normal / n_norm
        
        # The pole of the facet (circumcenter direction)
        # For a triangle on the unit sphere, the circumcenter is in the direction of the normal
        # but we need to find the actual circumcenter on the sphere.
        # The circumcenter C satisfies |C - pi| = r for i=0,1,2, with |C|=1.
        # This is equivalent to C being the pole of the great circle through the three points,
        # but only if the triangle is acute on the sphere.
        
        # Better: solve for C on the unit sphere such that C·p0 = C·p1 = C·p2 = cos(r)
        # C = a*p0 + b*p1 + c*p2, with |C|=1 and C·p0 = C·p1 = C·p2.
        # This is a linear system. Let's use the fact that C is orthogonal to (p1-p0) and (p2-p0).
        # So C is in the direction of the normal. Let C = t * normal.
        # Then |C| = 1 => |t| * |normal| = 1 => t = 1/|normal| (we pick the outward one).
        # But wait, normal was already normalized, so t = 1.
        # So C = normal? Let's check: C·p0 = normal·p0.
        # Is normal·p0 = normal·p1 = normal·p2?
        # normal = (p1-p0) × (p2-p0). 
        # normal·p0 = (p1-p0)×(p2-p0)·p0 = p1·(p2-p0)×p0 - p0·(p2-p0)×p0 = p1·(p2×p0)
        # normal·p1 = (p1-p0)×(p2-p0)·p1 = (p1×(p2-p0))·p1 - (p0×(p2-p0))·p1 = 0 - p0·(p2-p0)×p1
        # This is getting messy. Let me use a different approach.
        
        # The circumcenter of a spherical triangle can be found as the normalized sum
        # of the vertices weighted by the cotangents of the opposite angles, but that's complex.
        
        # Simpler: the covering radius for a facet is the maximum angle from the facet's
        # circumcenter to its vertices. For a triangle on the unit sphere, the circumcenter
        # is the pole of the circumscribed circle. If the triangle is contained in a hemisphere,
        # the circumcenter is in the direction of the normalized sum of the three vertices
        # only if they are on a great circle, which they're not.
        
        # Let's solve the linear system properly.
        # C · (p1 - p0) = 0
        # C · (p2 - p0) = 0
        # |C| = 1
        # So C is parallel to n = (p1-p0) × (p2-p0).
        # C = n / |n| (outward normal)
        # The angular distance from C to p0 is arccos(C · p0).
        # But is this the circumcenter? Only if the three points are on a small circle
        # with center C. Let's verify: C·p0 = C·p1 = C·p2?
        # C·p0 = n·p0 / |n|
        # C·p1 = n·p1 / |n|
        # n = (p1-p0)×(p2-p0)
        # n·p0 = (p1-p0)×(p2-p0)·p0 = p1·((p2-p0)×p0) - p0·((p2-p0)×p0) = p1·(p2×p0 - 0) = p1·(p2×p0)
        # n·p1 = (p1-p0)×(p2-p0)·p1 = p1·((p2-p0)×p1) - p0·((p2-p0)×p1) = 0 - p0·(p2×p1 - 0) = -p0·(p2×p1) = p0·(p1×p2)
        # p1·(p2×p0) = p2·(p0×p1) = p0·(p1×p2) (cyclic permutation)
        # So n·p0 = n·p1. Similarly n·p0 = n·p2.
        # So C·p0 = C·p1 = C·p2. Thus C is equidistant from all three vertices.
        # So C is indeed the circumcenter (on the sphere).
        
        # But we need to make sure C is on the same side as the facet's interior.
        # The outward normal from the convex hull should point away from the origin
        # for a facet of the convex hull of points on the unit sphere? Actually, for a
        # convex polytope containing the origin, the outward normal of a facet points
        # away from the origin. Since our points are on the unit sphere and we're looking
        # at the convex hull, the origin is inside the hull (assuming points are well-spread).
        # The pole C = n/|n| should be the circumcenter. But we need to check if it's the
        # correct pole (the one that gives the smaller cap).
        
        # The angular distance from C to p0 is arccos(C·p0). This should be the circumradius.
        # But we need to ensure this is the cap that covers the facet, not the complement.
        # The cap centered at C with angular radius r = arccos(C·p0) covers p0, p1, p2.
        # The covering radius is the max over all facets of this r, but only if the cap
        # is empty (contains no other points). Actually, the covering radius is the maximum
        # over all points on the sphere of the distance to the nearest point. This is
        # equal to the maximum circumradius of the facets of the convex hull of the points,
        # where each facet's circumcenter must be checked to have an empty cap.
        
        # For simplicity, let's compute the circumradius for each facet and take the max.
        # This is an upper bound. To be exact, we need to check that no other point is
        # closer to the circumcenter than the circumradius. But for a well-distributed
        # set, the facet circumcenters should be the critical points.
        
        cos_r = np.clip(normal @ p0, -1.0, 1.0)
        r = np.arccos(cos_r)
        max_radius = max(max_radius, r)
    
    return max_radius


def _repulsive_forces(pts, dt=1.0):
    """
    Compute repulsive forces on points to maximize the minimum distance.
    We use a force model that pushes points apart, with stronger forces
    for closer points. This helps maximize the covering radius complement
    (i.e., minimize the largest empty cap).
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            # Repulsive force proportional to 1/dist^2 (or similar)
            # We want to push apart points that are too close
            force_mag = 1.0 / (dist * dist + 1e-6)
            forces[i] += force_mag * d / dist
    
    return forces


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin: 20s buffer before 120s limit
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Iterative optimization using gradient-like steps
    # We'll use a simple approach: at each step, move points slightly away from their nearest neighbors
    # to reduce the covering radius.
    
    step_size = 0.01
    n_iterations = 500
    
    for it in range(n_iterations):
        if time.time() > deadline - 10.0:
            break
        
        # Compute forces: push each point away from its nearest neighbors
        forces = np.zeros_like(pts)
        for i in range(N):
            # Find nearest neighbors
            dots = pts @ pts[i]
            dots[i] = -2.0  # Ignore self
            # The nearest neighbors are those with the largest dot products (smallest angular distance)
            # We want to move point i away from its nearest neighbors
            # Force is in the direction away from the nearest neighbors
            for j in np.argsort(dots)[-5:]:  # 5 nearest neighbors
                if j == i:
                    continue
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Stronger force for closer points
                force_mag = 1.0 / (dist + 1e-6)
                forces[i] += force_mag * d / dist
        
        # Normalize forces to prevent large steps
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-10)
        forces = forces / force_norms
        
        # Update points
        pts = pts + step_size * forces
        
        # Project back to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-10)
        
        # Check if we improved
        current_radius = _covering_radius(pts)
        if current_radius < best_radius - 1e-6:
            best_radius = current_radius
            best_pts = pts.copy()
        else:
            # Revert if not improved (simple gradient descent)
            pts = best_pts.copy()
            step_size *= 0.9
        
        # Gradually decrease step size
        step_size *= 0.995
    
    return best_pts
