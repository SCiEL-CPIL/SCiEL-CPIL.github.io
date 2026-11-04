import numpy as np
import time
from scipy.spatial import ConvexHull

def _compute_circumradius(A, B, C):
    """
    Compute the angular circumradius of the spherical triangle with vertices A, B, C.
    A, B, C are unit vectors.
    """
    # Planar circumcenter logic
    # Vector normal to the plane defined by A, B, C
    # The circumcenter of the spherical triangle lies on the line through the origin 
    # and the planar circumcenter of the triangle ABC.
    
    # The planar circumcenter X satisfies X . (B-A) = 0 and X . (C-A) = 0
    # So X is parallel to (B-A) x (C-A)
    n = np.cross(B - A, C - A)
    if np.linalg.norm(n) < 1e-10:
        # Degenerate triangle, points are collinear or coincident
        # Return large value or distance to midpoint
        return np.pi / 2.0
        
    # Determine the scalar k such that X = k * n is equidistant from A and B
    # X . A = X . B  => k * n . A = k * n . B
    # This is always true if n is perpendicular to B-A.
    # We need X to be the circumcenter, so |X-A| = |X-B| = |X-C|
    # Actually, any point on the line is equidistant from A and B in the planar sense?
    # No. The condition X . A = X . B is satisfied by any X on the line.
    # We need to find the specific point.
    # The planar circumcenter is the intersection of the line n with the plane of the triangle?
    # No, the planar circumcenter is the point in the plane of ABC equidistant from A,B,C.
    # The line n passes through the origin. The plane of ABC does not necessarily pass through the origin.
    # The vector n is normal to the plane of ABC? No, n is normal to the vectors (B-A) and (C-A), so it is normal to the plane containing A,B,C.
    # So the line spanned by n is the normal to the plane.
    # The planar circumcenter is the projection of the origin onto the plane? No.
    # The planar circumcenter is the point X in the plane such that |X-A|=|X-B|=|X-C|.
    # Since X is in the plane, and n is normal to the plane, X = P + t*n? No.
    # Let's use the property that the circumcenter of the spherical triangle is the normalization of the vector:
    # V = A * (B x C) + B * (C x A) + C * (A x B) ? No.
    
    # Correct formula for circumcenter of spherical triangle:
    # The circumcenter O is given by:
    # O = normalize( A * sin(2*alpha) + B * sin(2*beta) + C * sin(2*gamma) ) ? No.
    
    # Let's use the linear system approach for the planar circumcenter.
    # Let X be the planar circumcenter.
    # X = A + s * (B-A) + t * (C-A)
    # We want |X-A|^2 = |X-B|^2 and |X-A|^2 = |X-C|^2
    # This is equivalent to X . (B-A) = 0.5 * (|B|^2 - |A|^2) = 0 (since unit)
    # And X . (C-A) = 0.
    # So X is orthogonal to B-A and C-A.
    # This implies X is parallel to n = (B-A) x (C-A).
    # So X = k * n.
    # We need to find k.
    # The condition X . A = X . B is automatically satisfied.
    # We need X to be the circumcenter, which means it is equidistant from A, B, C.
    # Since X is on the normal line, its distance to A is |X-A|.
    # We just need to pick the correct X.
    # Actually, the intersection of the normal line with the plane of the triangle is the circumcenter?
    # Yes, the circumcenter of a triangle in 3D space lies on the line perpendicular to the plane passing through the circumcenter of the 2D projection?
    # No, for a triangle in a plane not passing through the origin, the circumcenter is the unique point in the plane equidistant from vertices.
    # The vector n is normal to the plane.
    # The plane equation is n . x = d, where d = n . A.
    # The circumcenter X lies in the plane, so n . X = d.
    # Since X is parallel to n, X = k * n.
    # k * n . n = d => k * |n|^2 = d => k = d / |n|^2.
    # So X = n * (n . A) / |n|^2.
    
    d = np.dot(n, A)
    n_sq = np.dot(n, n)
    if n_sq < 1e-10:
        return np.pi / 2.0
    X = n * (d / n_sq)
    
    # X is the planar circumcenter.
    # The spherical circumcenter O is the normalization of X?
    # Yes, the line from the origin through the planar circumcenter intersects the sphere at the spherical circumcenter.
    O = X / np.linalg.norm(X)
    
    # Circumradius is the angle between O and A
    cos_r = np.clip(np.dot(O, A), -1.0, 1.0)
    r = np.arccos(cos_r)
    return r

def _compute_covering_radius(X):
    """
    Compute the covering radius of the configuration X.
    """
    try:
        hull = ConvexHull(X)
        max_rad = 0.0
        for simplex in hull.simplices:
            A = X[simplex[0]]
            B = X[simplex[1]]
            C = X[simplex[2]]
            r = _compute_circumradius(A, B, C)
            if r > max_rad:
                max_rad = r
        return max_rad
    except Exception:
        return np.pi

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 50
    
    # 1. Initial Guess: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Test Grid for Lloyd's Algorithm
    M = 5000
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    Y = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    best_X = X_fib
    best_radius = _compute_covering_radius(X_fib)
    
    # Multi-start strategies
    rng = np.random.default_rng(42)
    starts = [X_fib]
    # Add random perturbations
    for _ in range(5):
        X_pert = X_fib + rng.normal(0, 0.1, X_fib.shape)
        X_pert /= np.linalg.norm(X_pert, axis=1, keepdims=True)
        starts.append(X_pert)
        
    # Add a completely random start
    X_rand = rng.normal(0, 1, (N, 3))
    X_rand /= np.linalg.norm(X_rand, axis=1, keepdims=True)
    starts.append(X_rand)

    for X_init in starts:
        if time.time() > deadline:
            break
            
        X_curr = X_init.copy()
        
        # Lloyd's Algorithm
        for it in range(1000):
            if time.time() > deadline:
                break
                
            # 1. Assign Y to nearest X
            dots = Y @ X_curr.T
            dots = np.clip(dots, -1.0, 1.0)
            nearest_idx = np.argmax(dots, axis=1)
            
            # 2. Compute Centroids
            centroids = np.zeros_like(X_curr)
            for i in range(N):
                mask = nearest_idx == i
                if np.any(mask):
                    sum_y = np.sum(Y[mask], axis=0)
                    norm = np.linalg.norm(sum_y)
                    if norm > 1e-8:
                        centroids[i] = sum_y / norm
                    else:
                        centroids[i] = X_curr[i]
                else:
                    centroids[i] = X_curr[i]
            
            # 3. Update X
            # Damped update
            alpha = 1.0
            X_next = (1.0 - alpha) * X_curr + alpha * centroids
            X_next /= np.linalg.norm(X_next, axis=1, keepdims=True)
            
            # Check convergence
            diff = np.linalg.norm(X_next - X_curr)
            X_curr = X_next
            
            if diff < 1e-8:
                break
                
        # Evaluate exact covering radius
        radius = _compute_covering_radius(X_curr)
        if radius < best_radius:
            best_radius = radius
            best_X = X_curr.copy()
            
    # Final Polish
    # If we have time, we can try to refine best_X further
    # For now, return the best found
    return best_X
