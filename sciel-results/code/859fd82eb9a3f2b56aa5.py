"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """Compute the covering radius in radians for a set of points on the unit sphere."""
    points = np.array(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Build convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: brute force on a fine grid
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2*np.pi, 400)
        TH, PH = np.meshgrid(theta, phi)
        grid = np.column_stack([
            np.sin(TH)*np.cos(PH).ravel(),
            np.sin(TH)*np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        # For each grid point find nearest point
        dists = np.linalg.norm(grid[:, None, :] - points[None, :, :], axis=2)
        min_dists = dists.min(axis=1)
        # Convert chord distance to angle: angle = 2*arcsin(chord/2)
        angles = 2 * np.arcsin(np.clip(min_dists / 2.0, 0, 1))
        return angles.max()
    
    # For each facet, compute the circumcenter and the distance to the farthest vertex
    # The covering radius is the max over all facets of the angular distance from the facet's circumcenter to its vertices
    # But we need to check that the circumcenter lies within the cap (i.e., the facet is "empty")
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the vertices of this facet
        verts = points[simplex]
        
        # Compute the circumcenter of this spherical polygon (triangle or quadrilateral)
        # For a triangle on the sphere, the circumcenter is the normalized sum of the vertices
        # weighted by the sines of the opposite angles, but a simpler approach:
        # The circumcenter c satisfies |c - vi| = |c - vj| for all i, j
        # This means c is in the plane perpendicular to the edge midpoints... 
        # Actually, for a spherical triangle, the circumcenter is the point equidistant from all three vertices.
        # It can be found as the normalized cross product of (v1-v2) x (v1-v3) ... no.
        
        # Better approach: solve for c such that c·vi = c·vj for all i,j
        # i.e., c·(vi - vj) = 0
        # So c is orthogonal to all edge vectors, meaning c is parallel to the normal of the plane containing the vertices.
        # But that's just the centroid direction for coplanar points.
        
        # Actually, for points on the unit sphere, the circumcenter of a spherical triangle is the intersection of the sphere
        # with the line through the origin perpendicular to the plane of the triangle... no, that's only if the triangle is "small".
        
        # Let me use a different approach: the covering radius is the maximum over all points y on the sphere of min_i angle(y, xi).
        # The worst point y must be a vertex of the Voronoi diagram, which corresponds to a circumcenter of a Delaunay triangle.
        # For a set of points on the sphere, the Delaunay triangulation is the dual of the Voronoi diagram.
        # The circumcenter of a Delaunay triangle (xi, xj, xk) is the point c on the sphere equidistant from xi, xj, xk.
        
        # For three points on the unit sphere, the circumcenter c satisfies:
        # c·xi = c·xj = c·xk
        # This means c·(xi - xj) = 0 and c·(xi - xk) = 0
        # So c is orthogonal to (xi - xj) and (xi - xk), hence c is parallel to (xi - xj) × (xi - xk)
        
        v1, v2, v3 = verts[0], verts[1], verts[2]
        # Cross product of (v1-v2) and (v1-v3)
        e1 = v1 - v2
        e2 = v1 - v3
        c = np.cross(e1, e2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            continue
        c = c / c_norm
        
        # Check if c is on the same side as the facet normal (i.e., the cap is empty)
        # The facet normal points outward from the convex hull
        # We need to verify that c is actually a valid circumcenter (the cap centered at c with radius = angle(c, v1) contains no other points)
        
        # Compute the angular distance from c to each vertex
        d1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(c, v3), -1, 1))
        r = max(d1, d2, d3)
        
        # Verify that no other point is closer to c than r
        # (This check ensures the facet is "empty" in the Voronoi sense)
        if len(points) > 3:
            other_indices = [i for i in range(len(points)) if i not in simplex]
            if other_indices:
                other_dots = np.array([np.dot(c, points[i]) for i in other_indices])
                other_angles = np.arccos(np.clip(other_dots, -1, 1))
                if other_angles.min() < r - 1e-9:
                    # This circumcenter is not valid (another point is closer)
                    continue
        
        if r > max_radius:
            max_radius = r
    
    return max_radius


def _refine(points, max_time=110.0):
    """Refine the point configuration using local optimization."""
    points = np.array(points, dtype=float)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Use a simple iterative approach: for each point, try to move it to reduce the local covering radius
    # We'll use a gradient-free approach: for each point, compute the "worst" direction and move slightly in that direction
    
    lr = 0.05  # learning rate
    for iteration in range(2000):
        if time.time() > deadline - 10:
            break
        
        for i in range(N):
            if time.time() > deadline - 10:
                    break
            
            # Compute the covering radius contribution from point i
            # Find the point on the sphere farthest from the configuration, but only considering the neighborhood of point i
            # Simplified: compute the "gap" around point i
            
            # Find the nearest neighbors of point i
            dots = np.array([np.dot(points[i], points[j]) for j in range(N)])
            angles = np.arccos(np.clip(dots, -1, 1))
            # Sort by angle
            order = np.argsort(angles)
            # Get the 6 nearest neighbors (excluding self)
            neighbors = [order[j] for j in range(1, 7)]
            
            # The worst direction is roughly the bisector of the largest gap between neighbors
            # Compute angles between consecutive neighbors
            neighbor_angles = angles[neighbors]
            
            # Try a few candidate directions
            best_move = None
            best_score = np.inf
            
            # Candidate: move away from the centroid of neighbors
            neighbor_pts = points[neighbors]
            centroid = neighbor_pts.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                # Move in the opposite direction
                candidate = points[i] - lr * centroid
                candidate = candidate / np.linalg.norm(candidate)
                # Check if this improves things (simplified: just check it doesn't get too close to another point)
                min_dot = np.min(np.dot(candidate, points))
                if min_dot < 1.0 - 1e-6:
                    # Evaluate a quick metric
                    # Compute the max angle to nearest neighbor for all points
                    new_points = points.copy()
                    new_points[i] = candidate
                    dots_new = np.array([np.dot(new_points[j], new_points[k]) for j in range(N) for k in range(N)])
                    angles_new = np.arccos(np.clip(dots_new, -1, 1))
                    # For each point, find the max angle to its nearest other point
                    # This is a rough proxy for covering radius
                    max_min_angle = 0
                    for j in range(N):
                        row = angles_new[j*N:(j+1)*N]
                        row[j] = np.pi  # exclude self
                        max_min_angle = max(max_min_angle, row.min())
                    if max_min_angle < best_score:
                        best_score = max_min_angle
                        best_move = candidate
            
            # Also try moving in the direction of the worst gap
            # Find the largest angular gap between consecutive neighbors
            if len(neighbors) >= 2:
                # Compute the angles of neighbors around point i
                # Use a local coordinate system
                v1 = points[i]
                # Find a vector orthogonal to v1
                if abs(v1[0]) < 0.9:
                    u = np.cross(v1, [1, 0, 0])
                else:
                    u = np.cross(v1, [0, 1, 0])
                u = u / np.linalg.norm(u)
                w = np.cross(v1, u)
                
                # Project neighbors onto the tangent plane
                proj = []
                for j in neighbors:
                    p = points[j]
                    pu = np.dot(p, u)
                    pw = np.dot(p, w)
                    proj.append(np.arctan2(pw, pu))
                proj = np.array(proj)
                proj = (proj + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
                proj_sorted = np.sort(proj)
                
                # Find largest gap
                gaps = np.diff(proj_sorted)
                gaps = np.append(gaps, 2*np.pi - proj_sorted[-1] + proj_sorted[0])
                max_gap_idx = np.argmax(gaps)
                
                # The direction of the largest gap
                if max_gap_idx < len(proj_sorted) - 1:
                    angle1 = proj_sorted[max_gap_idx]
                    angle2 = proj_sorted[max_gap_idx + 1]
                else:
                    angle1 = proj_sorted[-1]
                    angle2 = proj_sorted[0] + 2 * np.pi
                
                mid_angle = (angle1 + angle2) / 2
                if angle2 < angle1:
                    mid_angle += np.pi
                
                # Convert back to 3D
                dir_vec = u * np.cos(mid_angle) + w * np.sin(mid_angle)
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                
                # Move in this direction
                candidate = points[i] + lr * dir_vec
                candidate = candidate / np.linalg.norm(candidate)
                
                # Quick check
                min_dot = np.min(np.dot(candidate, points))
                if min_dot < 1.0 - 1e-6:
                    new_points = points.copy()
                    new_points[i] = candidate
                    dots_new = np.array([np.dot(new_points[j], new_points[k]) for j in range(N) for k in range(N)])
                    angles_new = np.arccos(np.clip(dots_new, -1, 1))
                    max_min_angle = 0
                    for j in range(N):
                        row = angles_new[j*N:(j+1)*N]
                        row[j] = np.pi
                        max_min_angle = max(max_min_angle, row.min())
                    if max_min_angle < best_score:
                        best_score = max_min_angle
                        best_move = candidate
            
            if best_move is not None:
                points[i] = best_move
    
    return points


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine
    points = _refine(points, max_time=110.0)
    
    # Ensure points are on the unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
