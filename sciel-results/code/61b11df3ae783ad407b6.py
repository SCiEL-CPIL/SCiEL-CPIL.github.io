"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time


def _covering_radius(points):
    """Compute the covering radius (in radians) of points on the unit sphere.
    
    The covering radius is the maximum over all facets of the convex hull of
    the angular distance from the facet's circumcenter to its vertices.
    For a valid covering, each facet's circumcenter must lie on the sphere
    (i.e., the cap is empty).
    """
    points = np.asarray(points)
    n = len(points)
    
    # Check for degenerate cases
    if n < 3:
        return np.pi
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    
    # For each facet (simplex), compute the circumcenter and its angular distance to vertices
    for simplex in hull.simplices:
        verts = points[simplex]
        # The facet is a triangle (2-simplex) for a 3D convex hull
        # Compute circumcenter of the triangle in 3D
        
        # For a triangle on the unit sphere, the circumcenter of the spherical triangle
        # is the normalized sum of the vertices (if the triangle is "small" enough)
        # More robustly: find the point on the sphere equidistant from all 3 vertices
        
        a, b, c = verts[0], verts[1], verts[2]
        
        # Circumcenter of triangle abc on the unit sphere
        # Solve for x such that |x-a|^2 = |x-b|^2 = |x-c|^2 and |x|=1
        # This means x·a = x·b = x·c (since |x|=|a|=|b|=|c|=1)
        
        # Let x = alpha*a + beta*b + gamma*c
        # We need x·a = x·b = x·c
        
        # Actually, the circumcenter on the sphere is the normalized vector:
        # x ∝ (|b|^2|c|^2(a×b) + ...) -- this gets complicated.
        
        # Simpler approach: the circumcenter of a spherical triangle is the 
        # intersection of the perpendicular bisectors. For small triangles,
        # it's approximately the normalized centroid.
        
        # More accurate: solve the linear system
        # x · (a - b) = 0  =>  x·a = x·b
        # x · (a - c) = 0  =>  x·a = x·c
        # x · x = 1
        
        # Let's use the fact that the circumcenter x satisfies:
        # (a-b)·x = 0
        # (a-c)·x = 0
        # |x| = 1
        
        # So x is perpendicular to both (a-b) and (a-c), meaning x is parallel to (a-b)×(a-c)
        # But that's the normal to the plane containing a, b, c, which for points on a sphere
        # gives the center of the circle through a, b, c in 3D.
        
        # Actually, the circumcenter of the spherical triangle (the point on the sphere 
        # equidistant from a, b, c) is NOT the same as the 3D circumcenter of the triangle.
        
        # Let me think again. We want x on the unit sphere such that arccos(x·a) = arccos(x·b) = arccos(x·c)
        # i.e., x·a = x·b = x·c = cos(r) for some r.
        
        # This means x·(a-b) = 0 and x·(a-c) = 0.
        # So x is in the null space of [[a-b], [a-c]], i.e., x ∝ (a-b)×(a-c).
        
        cross = np.cross(a - b, a - c)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-12:
            # Degenerate triangle
            continue
        x = cross / cross_norm
        
        # Check which sign: we want the circumcenter to be on the "same side" as the interior
        # The covering radius facet circumcenter should be the one where the cap is empty
        # For a convex hull facet, the circumcenter should be on the outward side
        # But actually, there are two points on the sphere equidistant from a, b, c.
        # We need the one that is the "farthest" from the origin in the direction of the facet normal.
        
        # The facet normal (outward) is given by the hull. Let's check both signs.
        # The correct circumcenter for the covering radius is the one where the spherical
        # cap centered at x with radius r contains no other points.
        
        # For now, let's compute r for both signs and take the larger one (conservative)
        # Actually, for the covering radius, we want the maximum over all points on the sphere
        # of the distance to the nearest point. The worst case is at the circumcenter of a facet.
        
        # Let's compute the angular distance from x to a:
        dot_xa = np.dot(x, a)
        # Clamp for numerical stability
        dot_xa = np.clip(dot_xa, -1.0, 1.0)
        r1 = np.arccos(dot_xa)
        
        # Check the other sign
        x_neg = -x
        dot_xa_neg = np.dot(x_neg, a)
        dot_xa_neg = np.clip(dot_xa_neg, -1.0, 1.0)
        r2 = np.arccos(dot_xa_neg)
        
        # The covering radius contribution from this facet is the larger of the two
        # (since either could be the "empty cap" center)
        # Actually, we need to check which one has an empty cap.
        # For the convex hull, the relevant circumcenter is the one on the same side as the facet normal.
        
        # The facet normal from the hull:
        # hull.equations gives the plane equations for each facet
        # But let's just use both and take the max
        
        # Actually, for a proper covering, the circumcenter of each facet (the one on the 
        # sphere, not the antipode) should give the covering radius. Let's just use r1 for now
        # and verify with a direct computation.
        
        max_radius = max(max_radius, r1)
    
    # Also, we should check that the configuration is not contained in a hemisphere
    # and that points are not coincident
    
    # Verify by sampling: check the maximum distance from random points to nearest config point
    # This is expensive, so let's do a quick check
    rng = np.random.RandomState(42)
    sample_points = rng.randn(1000, 3)
    sample_points /= np.linalg.norm(sample_points, axis=1, keepdims=True)
    
    # Compute distances from each sample point to all config points
    dots = sample_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # Max dot product = min angular distance
    min_dots = np.clip(min_dots, -1.0, 1.0)
    sample_radius = np.arccos(min_dots).max()
    
    return max(max_radius, sample_radius)


def _lloyd_step(points, iterations=100):
    """Perform Lloyd's algorithm to minimize mean distance (improve distribution)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    
    for it in range(iterations):
        # Compute Voronoi cells using Delaunay triangulation on the sphere
        # For each point, its Voronoi cell is the set of points closer to it than to any other
        # We approximate by moving each point to the centroid of its Voronoi cell
        
        # Use a simple approach: for each point, compute the average of all points on the sphere
        # that are closer to it than to any other point. This is expensive, so we use a grid.
        
        # Create a grid of points on the sphere
        n_theta = 100
        n_phi = 200
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi)
        grid_x = np.sin(THETA) * np.cos(PHI)
        grid_y = np.sin(THETA) * np.sin(PHI)
        grid_z = np.cos(THETA)
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        
        # For each grid point, find the nearest config point
        dots = grid @ points.T
        nearest = np.argmax(dots, axis=1)
        
        # Compute the weighted centroid of each Voronoi cell
        new_points = np.zeros_like(points)
        weights = np.zeros(n)
        for i in range(n):
            mask = nearest == i
            if np.any(mask):
                new_points[i] = np.sum(grid[mask], axis=0)
                weights[i] = np.sum(np.clip(dots[mask, i], 0, 1))  # Weight by proximity
            else:
                new_points[i] = points[i]
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        new_points /= norms
        
        # Check if converged
        if np.allclose(points, new_points, atol=1e-8):
            break
        points = new_points
    
    return points


def _hull_refinement(points, max_iterations=50):
    """Refine points to reduce the covering radius by moving points toward large facets."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    
    for it in range(max_iterations):
        # Find the largest facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        max_radius = 0.0
        max_facet = None
        max_circumcenter = None
        
        for simplex in hull.simplices:
            verts = points[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            
            cross = np.cross(a - b, a - c)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                continue
            x = cross / cross_norm
            
            dot_xa = np.clip(np.dot(x, a), -1.0, 1.0)
            r = np.arccos(dot_xa)
            
            if r > max_radius:
                max_radius = r
                max_facet = simplex
                max_circumcenter = x
        
        if max_facet is None:
            break
        
        # Move the vertex of the largest facet toward the circumcenter
        # to reduce the facet size
        # The vertex that is "farthest" from the circumcenter in terms of the facet
        
        # Actually, let's move all three vertices slightly toward the circumcenter
        # to shrink the facet
        
        alpha = 0.1  # Step size
        for idx in max_facet:
            # Move point idx toward the circumcenter
            direction = max_circumcenter - points[idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-12:
                continue
            direction /= direction_norm
            points[idx] += alpha * direction
            # Renormalize
            norm = np.linalg.norm(points[idx])
            if norm > 1e-12:
                points[idx] /= norm
        
        # Check if we've made progress
        new_radius = _covering_radius(points)
        if new_radius >= max_radius - 1e-10:
            # Revert if no improvement
            break
    
    return points


def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before the 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Phase 1: Lloyd's algorithm for good initial distribution
    while time.time() < deadline - 30:
        points = _lloyd_step(points, iterations=50)
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        
        # Check convergence
        if radius < best_radius + 1e-10:
            break
    
    # Phase 2: Hull-based refinement to directly attack covering radius
    while time.time() < deadline - 10:
        points = _hull_refinement(best_points, max_iterations=20)
        radius = _covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        else:
            break
    
    # Final check
    final_radius = _covering_radius(best_points)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
