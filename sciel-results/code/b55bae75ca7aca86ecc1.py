"""Spherical Lloyd's algorithm (Voronoi relaxation) for covering the sphere with 50 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_centroid(facet_points):
    """Compute the spherical centroid (normalized mean) of a set of points on the sphere."""
    centroid = np.mean(facet_points, axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-12:
        return None
    return centroid / norm

def _compute_covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This equals the maximum circumradius of the
    facets of the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0  # Fallback: worst case
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # The circumcenter of a spherical triangle is the normalized sum of the vertices
        # (for equiangular-ish triangles this is a good approximation, but for exact circumradius
        # we need to find the point equidistant from all three vertices)
        
        # For a spherical triangle, the circumradius is the angular distance from the circumcenter
        # to any vertex. The circumcenter on the sphere is the normalized vector sum of the vertices
        # only if the triangle is acute. In general, we need to solve for the point equidistant
        # from all three vertices.
        
        # A simpler approach: the circumradius of a facet is the maximum angle from the facet's
        # circumcenter to its vertices. For a planar triangle (small facets), this is close to
        # the planar circumradius.
        
        # Actually, for the covering radius, we need the farthest point in each Voronoi cell.
        # The farthest point in a Voronoi cell on the sphere is the circumcenter of the corresponding
        # facet (if it lies within the facet's Voronoi cell region).
        
        # Let's compute the circumcenter of the spherical triangle properly.
        # The circumcenter is the point on the sphere equidistant from all three vertices.
        # This is the normalized cross product of the edge vectors, or equivalently,
        # the normalized sum of the vertices for small triangles.
        
        # More robust: solve for the point p such that p·a = p·b = p·c for unit vectors a, b, c.
        # This means p is in the null space of [a-b, a-c] (in the sense that (a-b)·p = 0 and (a-c)·p = 0)
        # So p is proportional to (a-b) × (a-c)
        
        a, b, c = facet_points[0], facet_points[1], facet_points[2]
        
        # Circumcenter direction
        circum_dir = np.cross(a - b, a - c)
        circum_norm = np.linalg.norm(circum_dir)
        
        if circum_norm < 1e-12:
            # Degenerate case
            continue
        
        circum_center = circum_dir / circum_norm
        
        # The circumradius is the angular distance from circum_center to any vertex
        # But we need to make sure the circumcenter is on the correct side (same hemisphere as the facet)
        # Check if circum_center is on the same side as the facet normal
        facet_normal = np.cross(b - a, c - a)
        facet_normal_norm = np.linalg.norm(facet_normal)
        if facet_normal_norm > 1e-12:
            facet_normal = facet_normal / facet_normal_norm
            if np.dot(circum_center, facet_normal) < 0:
                circum_center = -circum_center
        
        # Angular distance from circum_center to vertex a
        dot_val = np.dot(circum_center, a)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        angle = np.arccos(dot_val)
        
        if angle > max_circumradius:
            max_circumradius = angle
    
    return np.degrees(max_circumradius)

def _lloyd_step(points):
    """Perform one Lloyd's algorithm step: move each point to the centroid of its Voronoi cell."""
    new_points = np.zeros_like(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return points  # If hull fails, return original points
    
    # For each point, find its Voronoi cell (set of facets incident to that point)
    # and compute the centroid of the cell
    for i in range(len(points)):
        # Find all facets that contain point i
        incident_facets = []
        for simplex in hull.simplices:
            if i in simplex:
                incident_facets.append(simplex)
        
        if len(incident_facets) == 0:
            # Point is not on the hull (shouldn't happen for points on the sphere)
            new_points[i] = points[i]
            continue
        
        # The Voronoi cell of point i is the region of the sphere closest to i.
        # The centroid of the Voronoi cell can be approximated by the centroid of
        # all the vertices of the incident facets, weighted by facet area.
        
        # Better approach: the spherical Voronoi centroid is the normalized sum of
        # the facet circumcenters weighted by facet area, or simply the normalized
        # sum of all unique vertices of incident facets.
        
        # Simplest robust approach: collect all unique vertices from incident facets
        # and compute their normalized mean.
        vertex_set = set()
        for simplex in incident_facets:
            for v in simplex:
                vertex_set.add(v)
        
        if len(vertex_set) == 0:
            new_points[i] = points[i]
            continue
        
        # Weighted centroid: weight each vertex by the number of incident facets
        # it appears in (this approximates area weighting)
        vertex_counts = {}
        for simplex in incident_facets:
            for v in simplex:
                vertex_counts[v] = vertex_counts.get(v, 0) + 1
        
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        for v, count in vertex_counts.items():
            weighted_sum += count * points[v]
            total_weight += count
        
        if total_weight > 0:
            centroid = weighted_sum / total_weight
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
                new_points[i] = centroid / centroid_norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    
    return new_points

def construct():
    """Construct 50 points on the unit sphere using spherical Lloyd's algorithm."""
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Run Lloyd's algorithm iterations
    start_time = time.time()
    max_time = 110.0  # Leave some time margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for iteration in range(200):
        if time.time() - start_time > max_time:
            break
        
        new_points = _lloyd_step(points)
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check if points are distinct
        if np.any(np.linalg.norm(new_points[:, None, :] - new_points[None, :, :], axis=2) < 1e-10):
            # Points are too close, skip this step
            continue
        
        # Check if configuration is not in a closed hemisphere
        # (all points should span the sphere)
        # Simple check: the centroid should be near the origin for a good distribution
        centroid = np.mean(new_points, axis=0)
        if np.linalg.norm(centroid) > 0.5:
            # Configuration might be in a hemisphere, be cautious
            pass
        
        radius = _compute_covering_radius(new_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = new_points.copy()
        
        points = new_points
    
    return best_points
