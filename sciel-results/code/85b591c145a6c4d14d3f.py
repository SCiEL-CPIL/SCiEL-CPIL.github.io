"""
Strategy: Exact Delaunay Triangulation Optimization
The current program uses a sampling-based heuristic to estimate the covering radius and 
moves points toward farthest sample points. This is inefficient and imprecise.

The new strategy uses the fact that for points on a sphere, the covering radius is 
exactly the maximum circumradius of the facets (triangles) of the Delaunay triangulation.
We compute the Delaunay triangulation using scipy.spatial.Delaunay (which works on the 
projected 2D or 3D points, but for sphere we need to be careful. Actually, scipy.spatial.Delaunay 
works on Euclidean convex hulls. For points on a sphere, the Delaunay triangulation on the 
sphere is equivalent to the faces of the convex hull that are visible from the origin? 
No, the Delaunay triangulation on the sphere is the set of triangles such that their 
circumcircle (small circle) contains no other points. This is equivalent to the faces of 
the convex hull of the points if the points are in general position and the origin is inside 
the convex hull? Not exactly. 

Actually, the simplest robust way to get the "spherical Delaunay" is to use the convex hull 
of the points. The faces of the convex hull are triangles. For each triangular face, we can 
compute its circumradius on the sphere. The covering radius is the maximum of these circumradii.
Why? Because any point on the sphere is either inside a face's Voronoi cell (where the nearest 
point is one of the face's vertices) or on an edge. The farthest point in a Voronoi cell from 
the vertices is the circumcenter of the corresponding Delaunay triangle (if the triangle is 
acute on the sphere) or on the boundary. For spherical Delaunay, the circumcenter is the 
point equidistant to the 3 vertices. The distance from this circumcenter to the vertices is 
the circumradius. If the triangle is obtuse, the circumcenter is outside, but the Voronoi cell 
is unbounded in the Euclidean sense, but on the sphere, it's still bounded. The farthest point 
in the Voronoi cell of a vertex within a facet is actually the midpoint of the longest edge if 
the triangle is obtuse? No, for covering radius, we look at the empty circumcircle. 
The covering radius is the maximum over all Delaunay triangles of the distance from the 
circumcenter to the vertices, provided the circumcenter lies within the triangle. If the 
circumcenter lies outside, the maximum distance in that cell is determined by the edges. 
However, it is a known result that for spherical Delaunay triangulations, the covering radius 
is the maximum circumradius of the triangles, assuming the triangulation is such that all 
circumcenters lie within the triangles (acute triangles). If a triangle is obtuse, the 
circumcenter is outside, and the "empty cap" center is actually on the edge. But wait, the 
task says "largest circumradius of the facets of the convex hull, each facet checked to have 
an empty cap". This implies we should just compute the circumradius of each convex hull face 
and take the max. This is a standard approximation/measure used in these tasks.

So, the algorithm is:
1. Generate initial points (Fibonacci spiral or random).
2. Compute ConvexHull.
3. For each face (triangle) of the hull:
   a. Compute the circumcenter of the 3 vertices on the sphere.
   b. Compute the angular distance from the circumcenter to any of the 3 vertices.
   c. This is the circumradius of that facet.
4. The covering radius is the maximum of these circumradii.
5. Gradient: For the worst facet (max circumradius), move the 3 vertices of that facet 
   towards the circumcenter? Or away from each other? 
   Actually, to reduce the circumradius, we want to shrink the triangle. 
   The circumcenter is the point equidistant to the vertices. 
   If we move the vertices towards the circumcenter, the triangle shrinks and the circumradius decreases.
   So, for the worst facet, identify the 3 points. Compute their circumcenter C.
   For each of the 3 points p_i, add a gradient step: p_i += lr * (C - p_i) / ||C - p_i||.
   This moves the points towards the center of the worst empty cap.
   We can also apply smaller forces to other facets.

This is a structural change from "move towards farthest sample" to "move towards circumcenter of worst Delaunay triangle".
"""

import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
_DEADLINE = 120.0


def _normalize(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        n = np.linalg.norm(pts)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / n
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1e-12
    pts[mask] = pts[mask] / norms[mask, None]
    if np.any(~mask):
        pts[~mask] = np.array([1.0, 0.0, 0.0])
    return pts


def _spherical_circumradius(p1, p2, p3):
    """
    Compute the circumradius of a spherical triangle defined by 3 unit vectors.
    The circumcenter is the normalized sum of the 3 vectors? No, that's the centroid.
    The circumcenter C satisfies |C - p1| = |C - p2| = |C - p3| (Euclidean distance on sphere? 
    No, angular distance).
    Angular distance d(C, p) = arccos(C dot p).
    We want C such that C dot p1 = C dot p2 = C dot p3 = cos(R).
    This means C is orthogonal to (p1 - p2) and (p2 - p3).
    So C is parallel to (p1 - p2) x (p2 - p3).
    Let v = (p1 - p2) x (p2 - p3).
    C = v / ||v||.
    Then cos(R) = C dot p1.
    R = arccos(C dot p1).
    
    Wait, is this always the correct circumcenter?
    Yes, the point equidistant from p1, p2, p3 on the sphere lies on the normal to the plane 
    containing p1, p2, p3. The normal is p1 x p2 (if p1,p2,p3 are coplanar with origin? No, 
    they are on the sphere, so they define a plane not necessarily through origin).
    The vector (p1 - p2) x (p2 - p3) is normal to the plane containing p1, p2, p3.
    So C = normalize((p1 - p2) x (p2 - p3)) is the circumcenter.
    """
    v = np.cross(p1 - p2, p2 - p3)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        # Degenerate triangle, return large radius or 0?
        # If points are close, radius is small. If collinear, undefined.
        # For numerical stability, return 0 if very small, or handle carefully.
        # Actually, if norm_v is small, the points are nearly collinear, so the triangle is thin.
        # The circumradius might be large or small depending on geometry.
        # Let's just return 0 for now to avoid crashes, or a small value.
        return 0.0, np.zeros(3)
    
    c = v / norm_v
    # Ensure c is on the same side as the triangle interior?
    # The dot product c dot p1 should be positive for the "minor arc" circumcenter.
    # If c dot p1 < 0, we should flip c.
    if np.dot(c, p1) < 0:
        c = -c
    
    cos_r = np.dot(c, p1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    r = np.arccos(cos_r)
    return r, c


def _covering_radius_exact(pts):
    """
    Compute the exact covering radius using the convex hull faces.
    Returns (radius_in_degrees, worst_facet_indices, circumcenters).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback if hull fails (e.g., coplanar points)
        return 180.0, [], []
    
    faces = hull.simplices
    max_r = 0.0
    worst_face = None
    worst_c = None
    radii = []
    centers = []
    
    for face in faces:
        p1, p2, p3 = pts[face[0]], pts[face[1]], pts[face[2]]
        r, c = _spherical_circumradius(p1, p2, p3)
        radii.append(r)
        centers.append(c)
        if r > max_r:
            max_r = r
            worst_face = face
            worst_c = c
            
    return np.degrees(max_r), worst_face, centers


def _local_optimize_delaunay(pts, max_iters=1000, time_limit=None, tol=1e-6):
    """
    Optimize using Delaunay circumradius gradient.
    """
    pts = _normalize(pts).copy()
    start_time = time.time()
    best_pts = pts.copy()
    best_radius, _, _ = _covering_radius_exact(pts)
    
    lr = 0.02
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        radius, worst_face, _ = _covering_radius_exact(pts)
        
        if radius < best_radius - tol:
            best_radius = radius
            best_pts = pts.copy()
        else:
            # No improvement, decay learning rate
            lr = max(0.001, lr * 0.95)
            if lr <= 0.001 and it > 10:
                # Try to restart with slight perturbation if stuck?
                # Or just break if no improvement for a while.
                pass
            
        if worst_face is None:
            break
            
        p1, p2, p3 = pts[worst_face[0]], pts[worst_face[1]], pts[worst_face[2]]
        r, c = _spherical_circumradius(p1, p2, p3)
        
        # Move the 3 points towards the circumcenter c
        # This shrinks the triangle, reducing its circumradius.
        new_pts = pts.copy()
        for idx in worst_face:
            diff = c - pts[idx]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                new_pts[idx] = pts[idx] + lr * diff / diff_norm
                
        new_pts = _normalize(new_pts)
        
        # Check if new configuration is better
        new_radius, _, _ = _covering_radius_exact(new_pts)
        
        if new_radius < best_radius:
            pts = new_pts.copy()
            best_radius = new_radius
            best_pts = new_pts.copy()
            # Keep lr same or increase slightly if improving fast?
            # For now, keep stable.
        else:
            # Revert
            pts = best_pts.copy()
            lr = max(0.001, lr * 0.9)
            
    return best_pts


def construct():
    start_time = time.time()
    deadline = _DEADLINE - 5.0
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius, _, _ = _covering_radius_exact(pts)
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts
    n_attempts = 8
    for attempt in range(n_attempts):
        if time.time() - start_time > deadline:
            break
            
        if attempt == 0:
            current = pts.copy()
        else:
            # Random perturbation of best found so far, or fresh random?
            # Fresh random might find different basins.
            # Let's try a mix: 50% fresh random, 50% perturbed best.
            if rng.random() < 0.5:
                current = _normalize(rng.normal(0, 1, size=(N, 3)))
            else:
                current = _normalize(best_pts + rng.normal(0, 0.1, size=(N, 3)))
                
        remaining = deadline - (time.time() - start_time)
        if remaining < 10:
            break
            
        opt_pts = _local_optimize_delaunay(current, max_iters=2000, time_limit=remaining - 5)
        opt_radius, _, _ = _covering_radius_exact(opt_pts)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
            
    # Final polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        final_pts = _local_optimize_delaunay(best_pts, max_iters=3000, time_limit=remaining - 5)
        final_radius, _, _ = _covering_radius_exact(final_pts)
        if final_radius < best_radius:
            best_pts = final_pts.copy()
            
    return _normalize(best_pts)
