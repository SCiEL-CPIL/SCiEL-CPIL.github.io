"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid, return large value
    
    max_r = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]
        a, b, c = tri
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        centroid = (a + b + c) / 3.0
        if np.dot(centroid, n) < 0:
            n = -n
        # Compute max angular distance from n to the 3 vertices
        dots = np.clip(n @ tri.T, -1.0, 1.0)
        angles = np.arccos(dots)
        max_r = max(max_r, np.max(angles))
    return max_r

def _worst_facet_circumcenter(pts):
    """
    Find the facet with the largest circumradius and return:
    - the index of the facet (simplex)
    - the circumcenter point on the sphere
    - the circumradius (in radians)
    - the three vertex indices
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, 0.0, None
    
    max_r = 0.0
    best_simplex = None
    best_cc = None
    best_verts = None
    
    for simplex in hull.simplices:
        tri = pts[simplex]
        a, b, c = tri
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        centroid = (a + b + c) / 3.0
        if np.dot(centroid, n) < 0:
            n = -n
        
        # Compute max angular distance from n to the 3 vertices
        dots = np.clip(n @ tri.T, -1.0, 1.0)
        angles = np.arccos(dots)
        r = np.max(angles)
        
        if r > max_r:
            max_r = r
            best_simplex = simplex
            best_cc = n
            best_verts = simplex
    
    return best_simplex, best_cc, max_r, best_verts

def _move_vertex_toward_circumcenter(pts, vertex_idx, circumcenter, step):
    """
    Move a point on the sphere toward a target point using spherical linear interpolation.
    Returns the new point.
    """
    p = pts[vertex_idx]
    t = circumcenter
    
    # Compute the angle between p and t
    dot_val = np.clip(np.dot(p, t), -1.0, 1.0)
    theta = np.arccos(dot_val)
    
    if theta < 1e-12:
        return p.copy()
    
    # Slerp: move a fraction 'step' of the way from p to t
    # We want to move toward t, so we interpolate with parameter step
    # Actually, we want to move vertex toward circumcenter to reduce the circumradius
    # The circumcenter is the point equidistant from all three vertices of the facet
    # Moving a vertex toward the circumcenter should reduce the maximum distance
    
    # Use slerp: new_p = (sin((1-step)*theta)/sin(theta)) * p + (sin(step*theta)/sin(theta)) * t
    sin_theta = np.sin(theta)
    if sin_theta < 1e-12:
        return p.copy()
    
    new_p = (np.sin((1.0 - step) * theta) / sin_theta) * p + (np.sin(step * theta) / sin_theta) * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _worst_facet_shrink(pts, max_iters=5000, tol=1e-10, time_budget=80.0):
    """
    Deterministic iterative method: find worst facet, move its vertices toward circumcenter.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    current_r = _covering_radius(pts)
    
    for it in range(max_iters):
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Find worst facet
        simplex, cc, r, verts = _worst_facet_circumcenter(pts)
        if simplex is None:
            break
        
        # If the circumradius is very small, we're done
        if r < tol:
            break
        
        # Move each of the three vertices toward the circumcenter
        # Use a step size that depends on the current circumradius
        # Smaller steps for better convergence
        step = min(0.1, 0.5 * r)  # Move up to 50% of the way, capped at 0.1
        
        new_pts = pts.copy()
        for idx in verts:
            new_pts[idx] = _move_vertex_toward_circumcenter(pts, idx, cc, step)
        
        # Check if this improves the covering radius
        new_r = _covering_radius(new_pts)
        
        if new_r < current_r - 1e-14:
            pts = new_pts
            current_r = new_r
        else:
            # Try a smaller step
            step2 = step * 0.5
            new_pts2 = pts.copy()
            for idx in verts:
                new_pts2[idx] = _move_vertex_toward_circumcenter(pts, idx, cc, step2)
            new_r2 = _covering_radius(new_pts2)
            
            if new_r2 < current_r - 1e-14:
                pts = new_pts2
                current_r = new_r2
            else:
                # Can't improve this facet, try a much smaller step
                step3 = step * 0.1
                new_pts3 = pts.copy()
                for idx in verts:
                    new_pts3[idx] = _move_vertex_toward_circumcenter(pts, idx, cc, step3)
                new_r3 = _covering_radius(new_pts3)
                
                if new_r3 < current_r - 1e-14:
                    pts = new_pts3
                    current_r = new_r3
                else:
                    # No improvement possible with this facet, break to avoid cycling
                    break
    
    return pts, current_r

def _repulsion_refine(pts, max_iters=2000, time_budget=60.0):
    """
    Refine using a repulsion-based approach: points repel each other,
    but with attraction toward the worst facet circumcenter.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    current_r = _covering_radius(pts)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Compute repulsion forces
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Repulsive force inversely proportional to distance squared
                force = d / (dist * dist)
                forces[i] += force
        
        # Add attraction toward worst facet circumcenter
        simplex, cc, r, verts = _worst_facet_circumcenter(pts)
        if simplex is not None and cc is not None:
            for idx in verts:
                # Attract toward circumcenter
                forces[idx] += 0.5 * (cc - pts[idx])
        
        # Project forces to tangent plane and apply
        new_pts = pts.copy()
        for i in range(N):
            # Project force to tangent plane
            p = pts[i]
            f = forces[i]
            f_tangent = f - np.dot(f, p) * p
            f_norm = np.linalg.norm(f_tangent)
            if f_norm > 1e-15:
                f_tangent = f_tangent / f_norm
                # Small step
                step = 0.01
                new_pts[i] = pts[i] + step * f_tangent
                new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        
        new_r = _covering_radius(new_pts)
        if new_r < current_r - 1e-14:
            pts = new_pts
            current_r = new_r
        else:
            # Reduce step size
            step = 0.001
            new_pts2 = pts.copy()
            for i in range(N):
                p = pts[i]
                f = forces[i]
                f_tangent = f - np.dot(f, p) * p
                f_norm = np.linalg.norm(f_tangent)
                if f_norm > 1e-15:
                    f_tangent = f_tangent / f_norm
                    new_pts2[i] = pts[i] + step * f_tangent
                    new_pts2[i] = new_pts2[i] / np.linalg.norm(new_pts2[i])
            
            new_r2 = _covering_radius(new_pts2)
            if new_r2 < current_r - 1e-14:
                pts = new_pts2
                current_r = new_r2
            else:
                break
    
    return pts, current_r

def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _icosahedron_based_points():
    """Generate points based on icosahedron vertices and face centers."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # Start with icosahedron vertices (12 points)
    pts = verts.copy()
    
    # Add face centers and edge midpoints to get to 50 points
    # Icosahedron has 20 faces and 30 edges
    # We need 38 more points
    
    # Add face centers (20 points)
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(verts)
        for simplex in hull.simplices:
            center = np.mean(verts[simplex], axis=0)
            center = center / np.linalg.norm(center)
            pts = np.vstack([pts, center])
    except Exception:
        pass
    
    # If we still need more, add random points
    while len(pts) < N:
        rand_pt = np.random.randn(3)
        rand_pt = rand_pt / np.linalg.norm(rand_pt)
        pts = np.vstack([pts, rand_pt])
    
    pts = pts[:N]
    return _normalize(pts)

def _random_points(seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate multiple initial configurations
    initial_configs = [
        _fibonacci_points(),
        _icosahedron_based_points(),
        _random_points(42),
        _random_points(123),
        _random_points(777),
    ]
    
    best_pts = None
    best_val = np.pi
    
    # Try each initial configuration with different refinement strategies
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        current = init_pts.copy()
        current_val = _covering_radius(current)
        
        # Strategy 1: Worst-facet shrink
        if time.time() < deadline - 40:
            shrunk_pts, shrunk_val = _worst_facet_shrink(current, max_iters=3000, time_budget=25.0)
            if shrunk_val < best_val:
                best_pts = shrunk_pts
                best_val = shrunk_val
        
        # Strategy 2: Repulsion refine from the shrunk result or original
        base_pts = shrunk_pts if shrunk_val < current_val else current
        if time.time() < deadline - 20:
            rep_pts, rep_val = _repulsion_refine(base_pts, max_iters=2000, time_budget=20.0)
            if rep_val < best_val:
                best_pts = rep_pts
                best_val = rep_val
        
        # Strategy 3: Another round of worst-facet shrink
        if time.time() < deadline - 10:
            final_pts, final_val = _worst_facet_shrink(best_pts if best_pts is not None else base_pts, 
                                                        max_iters=2000, time_budget=10.0)
            if final_val < best_val:
                best_pts = final_pts
                best_val = final_val
    
    # Final polish: very fine worst-facet shrink
    if time.time() < deadline - 5 and best_pts is not None:
        polished_pts, polished_val = _worst_facet_shrink(best_pts, max_iters=5000, tol=1e-12, time_budget=8.0)
        if polished_val < best_val:
            best_pts = polished_pts
            best_val = polished_val
    
    # Ensure validity
    if best_pts is None:
        best_pts = _fibonacci_points()
    
    best_pts = _normalize(best_pts)
    
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                best_pts[j] = best_pts[j] + 1e-6 * np.random.randn(3)
                best_pts = _normalize(best_pts)
    
    # Check if all points are in a closed hemisphere by checking if origin is inside convex hull
    try:
        hull = ConvexHull(best_pts)
        # Check if origin is inside: all facet normals (outward) should have positive dot with origin
        # Actually, for a convex polytope containing origin, the origin is a convex combination of vertices
        # Simple check: if the minimum distance from origin to any facet is > 0, origin is inside
        # Or just check that no hemisphere contains all points
        # A simpler heuristic: check that the points span all octants reasonably
        # If the centroid is close to origin, it's likely fine
        centroid = np.mean(best_pts, axis=0)
        if np.linalg.norm(centroid) > 0.5:
            # Might be in a hemisphere, try to fix
            # Reflect some points to balance
            for i in range(N):
                if np.dot(best_pts[i], centroid) < 0:
                    # Keep as is
                    pass
                else:
                    # Consider reflecting
                    pass
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
