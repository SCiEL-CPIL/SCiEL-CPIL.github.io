import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def compute_covering_radius(points):
    """
    Computes the covering radius in radians.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point.
    This is equivalent to the maximum circumradius of the spherical Delaunay triangles.
    """
    if len(points) < 3:
        return np.pi / 2
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Calculate convex hull of the 3D points
    # The faces of the convex hull correspond to the spherical Delaunay triangulation
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g., all points coplanar, though unlikely on sphere)
        # Estimate radius by max distance to nearest point on a grid? 
        # For robustness, just return a large value if hull fails
        return np.pi / 2

    max_radius = 0.0
    # Hull faces are indices of points
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        # Calculate circumcenter of the triangle on the sphere
        # The circumcenter in 3D is the intersection of the plane of the triangle with the line through the origin perpendicular to the plane?
        # No, the circumcenter of a spherical triangle is the intersection of the great circles perpendicular to the sides?
        # Actually, the circumcenter of the spherical triangle is the point on the sphere equidistant from the vertices.
        # This is the direction of the vector c = p0 + p1 + p2 (for small triangles) or more accurately:
        # The circumcenter is the normalized cross product of the normals of the edges?
        
        # A robust way: The circumcenter C is parallel to the vector:
        # c = (p0 x p1) x (p1 x p2) + ... ? 
        # Actually, the circumcenter of a spherical triangle is the intersection of the sphere with the line through the origin and the 3D circumcenter of the planar triangle formed by the chords? No.
        
        # The circumcenter of a spherical triangle is the unit vector:
        # c = (p0 * sin(A) + p1 * sin(B) + p2 * sin(C)) ? No.
        
        # Let's use the property that the circumcenter is equidistant from p0, p1, p2.
        # Let c be the circumcenter. c . p0 = c . p1 = c . p2 = cos(r).
        # This implies c is in the span of p0, p1, p2.
        # c = a*p0 + b*p1 + c*p2.
        # This is a linear system.
        
        # Simpler: The circumradius is the maximum of the angular distances from the circumcenter to the vertices.
        # The circumcenter is the normalized sum of the vertices weighted by the sines of the opposite angles?
        
        # Alternative approach: 
        # The circumradius r is the smallest radius such that a cap of radius r centered at some point on the sphere contains the three vertices.
        # This is equivalent to the circumradius of the planar triangle formed by the points projected onto the tangent plane at the circumcenter.
        
        # Let's calculate the 3D circumcenter of the triangle formed by the points p0, p1, p2 in R3.
        # The circumcenter O of the planar triangle p0, p1, p2 lies in the plane of the triangle.
        # The spherical circumcenter is the normalization of O? 
        # YES. The spherical circumcenter is the direction of the 3D circumcenter of the chordal triangle.
        
        # Calculate 3D circumcenter of triangle p0, p1, p2
        # Solve for O = p0 + u*(p1-p0) + v*(p2-p0)
        # (O-p0).(O-p0) = (O-p1).(O-p1)
        # (O-p0).(O-p0) = (O-p2).(O-p2)
        
        # Let A = p1 - p0, B = p2 - p0
        # O = p0 + u*A + v*B
        # |p0 + uA + vB - p0|^2 = |p0 + uA + vB - (p0+A)|^2
        # |uA + vB|^2 = |uA + vB - A|^2
        # |uA + vB|^2 = |uA + vB|^2 - 2(uA+vB).A + |A|^2
        # 2(uA+vB).A = |A|^2
        # Similarly 2(uA+vB).B = |B|^2
        
        # Matrix form:
        # [A.A, A.B] [u]   [0.5 A.A]
        # [B.A, B.B] [v] = [0.5 B.B]
        
        A = p1 - p0
        B = p2 - p0
        A_A = np.dot(A, A)
        A_B = np.dot(A, B)
        B_B = np.dot(B, B)
        
        # Solve 2x2 system
        det = A_A * B_B - A_B * A_B
        if abs(det) < 1e-12:
            # Degenerate triangle, skip or handle carefully
            # Use max distance to nearest vertex as a rough estimate?
            # For degenerate, the circumradius is effectively the max distance to the furthest point from the "center" (midpoint of longest side)
            # But let's just skip this face or use a conservative estimate
            # Actually, if degenerate, the points are nearly collinear. The circumradius is large.
            # Let's approximate by the max distance from the midpoint of the longest edge to the third point.
            # But for covering radius, we want the max. 
            # If det is small, the triangle is flat.
            # We'll just use the max distance from the centroid to the vertices as a proxy?
            # Better: if degenerate, the circumcenter is at infinity. The covering radius contribution is effectively the distance to the furthest point from the "center" of the cluster.
            # In the context of the hull, if a face is degenerate, it's a numerical issue.
            # Let's just skip it if it's too small, or use the max angular distance from the average vector.
            centroid = (p0 + p1 + p2) / 3.0
            centroid = centroid / np.linalg.norm(centroid)
            d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
            d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
            r_face = max(d0, d1, d2)
        else:
            u = (0.5 * A_A * B_B - 0.5 * A_B * B_B) / det
            v = (0.5 * A_B * A_A - 0.5 * A_A * B_B) / det
            
            O = p0 + u * A + v * B
            
            # Normalize O to get the spherical circumcenter
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-12:
                # O is near origin, means the triangle is right-angled or larger?
                # If O is origin, the circumcenter is undefined or at infinity?
                # Actually if O is origin, the points are 180 degrees apart?
                # Let's use the centroid fallback
                centroid = (p0 + p1 + p2) / 3.0
                centroid = centroid / np.linalg.norm(centroid)
                d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                r_face = max(d0, d1, d2)
            else:
                C = O / norm_O
                # Calculate angular distance from C to vertices
                d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
                r_face = max(d0, d1, d2)
        
        if r_face > max_radius:
            max_radius = r_face
            
    return max_radius

def energy_and_grad(points, power=4):
    """
    Calculate energy and gradient for potential V = sum 1/d^power
    """
    N_pts = len(points)
    # Compute pairwise differences
    # points shape (N, 3)
    # diff shape (N, N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-12] = 1e-12
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal
    mask = ~np.eye(N_pts, dtype=bool)
    
    # Energy
    # V = sum_{i<j} 1/d^power
    # Sum over all i!=j is 2 * sum_{i<j}
    # So we can sum all and divide by 2
    inv_dist_pow = 1.0 / (dist_sq ** (power / 2.0))
    inv_dist_pow[~mask] = 0.0
    energy = 0.5 * np.sum(inv_dist_pow)
    
    # Gradient
    # d/dp_i (1/d^power) = -power * (p_i - p_j) / d^(power+2)
    # Sum over j
    # grad_i = -power * sum_j (p_i - p_j) / d_ij^(power+2)
    
    # dist_sq shape (N, N)
    # We need (p_i - p_j) / d_ij^(power+2)
    # diff is (p_i - p_j)
    # coeff shape (N, N)
    coeff = power * (dist_sq ** (-(power + 2) / 2.0))
    coeff[~mask] = 0.0
    
    # grad_i = - sum_j coeff[i,j] * diff[i,j]
    # diff[i,j] = p_i - p_j
    # So grad = - sum_j coeff[i,j] * (p_i - p_j)
    # grad = - (sum_j coeff[i,j] * p_i) + (sum_j coeff[i,j] * p_j)
    # sum_j coeff[i,j] is a scalar for each i
    sum_coeff = np.sum(coeff, axis=1)
    # term1 = sum_coeff[:, np.newaxis] * points
    term1 = sum_coeff[:, np.newaxis] * points
    # term2 = sum_j coeff[i,j] * p_j
    # This is coeff @ points
    term2 = coeff @ points
    
    grad = -term1 + term2
    
    # Project gradient onto tangent space
    # g_tan = grad - (grad . p_i) p_i
    dot_prod = np.sum(grad * points, axis=1, keepdims=True)
    grad_tan = grad - dot_prod * points
    
    return energy, grad_tan

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_radius = np.pi / 2.0
    
    # Initializations
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [fib_points]
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Uniform on sphere
        vecs = rng.normal(size=(N, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        inits.append(vecs)
        
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        points = init_pts.copy()
        
        # Gradient Descent
        # Learning rate schedule
        lr = 0.05
        min_lr = 1e-6
        decay = 0.995
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            energy, grad = energy_and_grad(points, power=4)
            
            # Update
            points = points + lr * grad
            # Normalize
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            lr *= decay
            
            # Check for convergence
            if lr < min_lr:
                break
        
        # Evaluate covering radius
        radius = compute_covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        # Optional: Local refinement on the best facet
        # If time permits, do a small local search
        if time.time() < deadline - 10.0:
             # Simple local search: perturb points and see if radius decreases
             # This is expensive, so maybe just do it on the best candidate found so far
             pass

    if best_points is None:
        return fib_points
        
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def compute_covering_radius(points):
    """
    Computes the covering radius in radians.
    """
    if len(points) < 3:
        return np.pi / 2
    
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2

    max_radius = 0.0
    for face in hull.simplices:
        p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
        
        A = p1 - p0
        B = p2 - p0
        A_A = np.dot(A, A)
        A_B = np.dot(A, B)
        B_B = np.dot(B, B)
        
        det = A_A * B_B - A_B * A_B
        if abs(det) < 1e-12:
            centroid = (p0 + p1 + p2) / 3.0
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                r_face = np.pi / 2
            else:
                centroid = centroid / norm_c
                d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                r_face = max(d0, d1, d2)
        else:
            u = (0.5 * A_A * B_B - 0.5 * A_B * B_B) / det
            v = (0.5 * A_B * A_A - 0.5 * A_A * B_B) / det
            O = p0 + u * A + v * B
            norm_O = np.linalg.norm(O)
            if norm_O < 1e-12:
                centroid = (p0 + p1 + p2) / 3.0
                norm_c = np.linalg.norm(centroid)
                if norm_c < 1e-12:
                    r_face = np.pi / 2
                else:
                    centroid = centroid / norm_c
                    d0 = np.arccos(np.clip(np.dot(centroid, p0), -1, 1))
                    d1 = np.arccos(np.clip(np.dot(centroid, p1), -1, 1))
                    d2 = np.arccos(np.clip(np.dot(centroid, p2), -1, 1))
                    r_face = max(d0, d1, d2)
            else:
                C = O / norm_O
                d0 = np.arccos(np.clip(np.dot(C, p0), -1, 1))
                d1 = np.arccos(np.clip(np.dot(C, p1), -1, 1))
                d2 = np.arccos(np.clip(np.dot(C, p2), -1, 1))
                r_face = max(d0, d1, d2)
        
        if r_face > max_radius:
            max_radius = r_face
            
    return max_radius

def energy_and_grad(points, power=4):
    N_pts = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    mask = ~np.eye(N_pts, dtype=bool)
    
    inv_dist_pow = 1.0 / (dist_sq ** (power / 2.0))
    inv_dist_pow[~mask] = 0.0
    energy = 0.5 * np.sum(inv_dist_pow)
    
    coeff = power * (dist_sq ** (-(power + 2) / 2.0))
    coeff[~mask] = 0.0
    
    sum_coeff = np.sum(coeff, axis=1)
    term1 = sum_coeff[:, np.newaxis] * points
    term2 = coeff @ points
    
    grad = -term1 + term2
    
    dot_prod = np.sum(grad * points, axis=1, keepdims=True)
    grad_tan = grad - dot_prod * points
    
    return energy, grad_tan

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_radius = np.pi / 2.0
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [fib_points]
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        vecs = rng.normal(size=(N, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        inits.append(vecs)
        
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        points = init_pts.copy()
        lr = 0.05
        min_lr = 1e-6
        decay = 0.995
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            energy, grad = energy_and_grad(points, power=4)
            points = points + lr * grad
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            lr *= decay
            if lr < min_lr:
                break
        
        radius = compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()

    if best_points is None:
        return fib_points
        
    return best_points
# EVOLVE-BLOCK-END
