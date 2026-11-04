"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius_deg(points):
    """Compute the covering radius in degrees for a set of unit points."""
    if len(points) < 4:
        return 180.0
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0
    max_ang = 0.0
    for face in hull.simplices:
        # The facet is defined by points[face]. The normal pointing outward is
        # the normalized sum of the vertices (for a convex polytope containing origin).
        # Actually, for a convex hull of points on the unit sphere, the outward normal
        # of a facet can be computed as the normalized cross product or just the
        # normalized sum of the vertices if they are roughly coplanar and close.
        # A more robust way: the circumscribed sphere of the facet.
        # The center of the circumscribed sphere of a triangle (a, b, c) on the unit sphere
        # lies along the normal. The angular distance from the origin to the facet plane
        # is the covering radius contribution for that facet.
        # Let n be the unit normal to the facet pointing outward.
        # The distance from origin to the plane is d = n . a (since |a|=1, this is cos(theta)).
        # The angular distance is arccos(d).
        a = points[face[0]]
        b = points[face[1]]
        c = points[face[2]]
        # Normal vector
        v1 = b - a
        v2 = c - a
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            continue
        n = n / norm_n
        # Ensure n points outward (away from origin)
        # The origin is inside the hull, so n . a should be positive for outward normal?
        # Wait, a is on the surface. The vector from origin to a is a.
        # If n is outward, then the plane is n . x = n . a.
        # Since origin is inside, n . 0 = 0 < n . a (assuming n . a > 0).
        # So n . a should be positive. If not, flip n.
        if np.dot(n, a) < 0:
            n = -n
        d = np.dot(n, a)
        # d = cos(theta), where theta is the angle from the z-axis (normal) to the point a.
        # The angular distance from the origin (center of sphere) to the facet plane is arccos(d).
        # This is the angular radius of the cap covering this facet.
        theta = np.arccos(np.clip(d, -1.0, 1.0))
        if theta > max_ang:
            max_ang = theta
    return np.degrees(max_ang)

def _refine(points, max_iter=2000, tol=1e-10):
    """
    Refine points to minimize the covering radius using a subgradient-like method.
    We use a simple iterative scheme: move the point that defines the largest facet
    slightly towards the circumcenter of that facet, or adjust points to equalize facet radii.
    """
    points = points.copy()
    best_cr = _covering_radius_deg(points)
    
    # Simple local optimization:
    # 1. Compute covering radius and identify worst facet.
    # 2. Move points in the worst facet slightly to reduce that facet's radius.
    # This is tricky. A better approach:
    # Use a gradient-free method like Nelder-Mead on a smoothed objective?
    # Or use the fact that the optimal configuration has equal facet radii (for many).
    
    # Let's try a simpler approach: random perturbations + acceptance if CR decreases.
    # This is slow but robust.
    # To speed it up, we can use the "worst facet" gradient.
    
    # Let's implement a basic projected gradient step on the sphere.
    # Objective: f(x) = max_i arccos(n_i . x_i) ? No, it's max over facets.
    # Subgradient of max is the subgradient of the argmax term.
    
    # For now, let's use a simple hill-climbing with random restarts.
    # Given the time limit, we need to be efficient.
    
    # Let's try to use the convex hull to compute the "pressure" on each point.
    # Points that are vertices of large facets should be moved.
    
    # Actually, a very effective method for spherical codes is the "repulsive" force model.
    # But for covering, we want to minimize the max distance.
    
    # Let's stick to the subgradient idea.
    # At each step:
    # 1. Find the facet with the largest angular radius.
    # 2. Compute the subgradient for the points involved in that facet.
    # 3. Update the points by moving them slightly in the direction of the subgradient.
    
    # This is complex to implement correctly without errors.
    # Let's use a simpler heuristic:
    # If a facet is too large, move its vertices slightly towards the circumcenter of that facet.
    
    # Circumcenter of a spherical triangle is the normalized sum of the vertices? No.
    # It's the point on the sphere equidistant to the three vertices.
    # This is the pole of the plane containing the triangle.
    
    # Let's try this:
    for it in range(max_iter):
        if time.time() - start_time > 110:
            break
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        max_ang = 0.0
        worst_face = None
        for face in hull.simplices:
            a = points[face[0]]
            b = points[face[1]]
            c = points[face[2]]
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue
            n = n / norm_n
            if np.dot(n, a) < 0:
                n = -n
            d = np.dot(n, a)
            theta = np.arccos(np.clip(d, -1.0, 1.0))
            if theta > max_ang:
                max_ang = theta
                worst_face = face
        
        if worst_face is None:
            break
        
        # Move the points in the worst face towards the circumcenter
        # The circumcenter is the normalized n (if n is the outward normal, the circumcenter is -n? No.)
        # The facet is at distance d from origin along n. The "center" of the cap is n.
        # We want to reduce the distance d, i.e., increase n . a.
        # This means moving a, b, c towards n.
        
        a = points[worst_face[0]]
        b = points[worst_face[1]]
        c = points[worst_face[2]]
        
        # Compute the circumcenter direction (which is n)
        v1 = b - a
        v2 = c - a
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            break
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        
        # Move a, b, c slightly towards n
        step = 0.01
        for idx in worst_face:
            p = points[idx]
            # Project p onto the sphere, then move towards n
            # p_new = normalize(p + step * (n - p)) ?
            # Or simply p_new = normalize(p + step * n)
            p_new = p + step * n
            p_new = p_new / np.linalg.norm(p_new)
            points[idx] = p_new
        
        new_cr = _covering_radius_deg(points)
        if new_cr < best_cr - 1e-10:
            best_cr = new_cr
        else:
            # Revert
            points[worst_face[0]] = a
            points[worst_face[1]] = b
            points[worst_face[2]] = c
            # Try a smaller step or break
            step /= 2
            if step < 1e-6:
                break
    
    return points, best_cr

def construct():
    global start_time
    start_time = time.time()
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine
    points, best_cr = _refine(points, max_iter=500, tol=1e-10)
    
    # Check time
    if time.time() - start_time > 115:
        return points
    
    # Try a few random restarts if time permits
    for restart in range(3):
        if time.time() - start_time > 115:
            break
        # Perturb the current best
        perturb = points + 0.1 * np.random.randn(N, 3)
        perturb = perturb / np.linalg.norm(perturb, axis=1, keepdims=True)
        new_points, new_cr = _refine(perturb, max_iter=100, tol=1e-10)
        if new_cr < best_cr:
            points = new_points
            best_cr = new_cr
    
    return points
# EVOLVE-BLOCK-END
