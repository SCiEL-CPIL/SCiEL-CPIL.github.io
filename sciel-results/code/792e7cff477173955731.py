"""
Optimized 50-point spherical covering via convex hull facet circumradii and L-BFGS.

Strategy:
1. Compute exact covering radius as max circumradius of convex hull facets.
2. Minimize a soft-max (log-sum-exp) approximation of this radius using L-BFGS.
3. Multiple restarts from different initializations (Fibonacci, random, perturbed).
4. Final polish with Spherical Lloyd's algorithm.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _compute_facet_circumradii(pts):
    """
    Compute circumradii for all facets of the convex hull of pts.
    Returns (max_radius, all_radii, hull) or (None, None, None) if hull fails.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return 0.0, np.array([]), hull
        
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Vectorized circumcenter calculation
    # The circumcenter of a spherical triangle (a,b,c) is the unit vector u
    # such that u.a = u.b = u.c.
    # This is equivalent to finding the intersection of the sphere with the
    # plane defined by the linear system, or using the formula:
    # u = normalize( a * sin(A) + b * sin(B) + c * sin(C) )
    # where A, B, C are the angles opposite to a, b, c.
    # Alternatively, a more robust algebraic method:
    # The circumcenter is proportional to the cross product of the edges in a specific way,
    # but for spherical geometry, it's easier to solve the linear system:
    # u . a = u . b = u . c = cos(R)
    
    # Let's use the property that the circumcenter lies in the span of a,b,c.
    # Actually, a simpler vectorized approach for unit vectors:
    # The normal to the plane containing a,b,c is n = (a-b) x (b-c).
    # The circumcenter u is in the direction of the vector v such that v is equidistant.
    # For a triangle on a sphere, the circumcenter direction is given by:
    # v = a * |b-c|^2 * (a.(b-c)) ... no, this is getting complex.
    
    # Let's use the standard formula for spherical circumradius.
    # Side lengths:
    # ab = arccos(a.b), bc = arccos(b.c), ca = arccos(c.a)
    # The circumradius R is given by the formula for spherical triangles.
    # However, a numerical linear solve per facet is safer and fast enough for ~100 facets.
    # But we want vectorization.
    
    # Alternative: The circumcenter u satisfies:
    # u . a = u . b = u . c
    # This implies u is orthogonal to (a-b) and (b-c).
    # So u is parallel to (a-b) x (b-c).
    # Let n = (a-b) x (b-c).
    # Then u = ± normalize(n).
    # We need to choose the sign such that u points "outward" or towards the interior?
    # Actually, for the circumradius, we just need the distance.
    # The distance from u to a is arccos(u.a).
    # Since u is orthogonal to a-b, u.a = u.b.
    # So we just compute R = arccos(u.a).
    
    ab = a - b
    bc = b - c
    n = np.cross(ab, bc)
    
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    valid = norm_n[:, 0] > 1e-12
    norm_n[~valid] = 1.0
    u = n / norm_n
    
    # Check orientation: u should be such that the triangle is "visible" from u?
    # Actually, for the circumradius of the facet, we want the radius of the cap covering the facet.
    # The circumcenter is the center of the smallest cap containing the 3 vertices.
    # The vector u computed above is normal to the plane of the triangle.
    # There are two such directions. One points towards the "pole" of the triangle.
    # The distance to the vertices is the same for both ±u.
    # So R = arccos(|u.a|)? No.
    # If u is the circumcenter, u.a = cos(R).
    # So R = arccos(u.a).
    # We need to ensure u is the correct circumcenter.
    # The circumcenter is the point equidistant from a,b,c.
    # The vector n = (a-b)x(b-c) is perpendicular to the plane.
    # The circumcenter lies on the line passing through the origin and parallel to n?
    # No, the circumcenter of a spherical triangle is the intersection of the sphere
    # and the line passing through the origin and the circumcenter of the planar triangle?
    # No.
    
    # Let's stick to the linear system solution for robustness, vectorized.
    # We want u such that u.a = u.b = u.c.
    # This means u is orthogonal to a-b and b-c.
    # So u is parallel to (a-b) x (b-c).
    # Let u0 = normalize((a-b) x (b-c)).
    # Then u0.a = u0.b = u0.c = k.
    # The circumradius is arccos(|k|)?
    # If the triangle is "small", k > 0.
    # If the triangle is "large" (> hemisphere), the circumcenter might be on the other side.
    # However, for a convex hull of points on a sphere, the facets are typically small enough
    # that the circumcenter is on the "outside" of the convex hull?
    # Actually, the covering radius is the max distance to the *nearest* point.
    # The worst case is the circumcenter of a facet.
    # The distance from the circumcenter to the vertices is the circumradius.
    
    # Let's compute dot products
    dots = np.sum(u * a, axis=1)
    # The circumradius is arccos(dots).
    # We need to be careful with the sign.
    # If the triangle contains the origin in its convex hull (which it doesn't, it's on the sphere),
    # this would be ambiguous.
    # But for convex hull facets, the normal n points outward.
    # The circumcenter is in the direction of n if the triangle is acute?
    # Let's just take the absolute value of the dot product?
    # No, if the circumcenter is on the opposite side, the distance is arccos(-dot).
    # The covering radius is the max over all facets.
    # The circumcenter of a facet is the point on the sphere furthest from the 3 vertices?
    # No, it's the point equidistant.
    # There are two points equidistant from 3 points on a sphere (antipodal).
    # One is the circumcenter of the triangle, the other is its antipode.
    # The circumradius is the distance to the circumcenter.
    # We want the circumcenter that is "close" to the vertices.
    # So we want arccos(max(dot, -dot))?
    # Actually, the circumradius R satisfies sin(R) = ...
    # Let's use the formula:
    # cos(R) = (1 + 2 cos(A)cos(B)cos(C) - cos^2(A) - cos^2(B) - cos^2(C)) / ...
    # This is complicated.
    
    # Simpler: The circumcenter u is the normalized sum of the vertices weighted by sines of opposite angles.
    # Or, we can just solve the linear system for u.
    # u = alpha*a + beta*b + gamma*c
    # u.a = u.b = u.c
    # This is a 3x3 linear system for alpha, beta, gamma.
    # Let's do this vectorized.
    
    # Gram matrix G = [[1, a.b, a.c], [a.b, 1, b.c], [a.c, b.c, 1]]
    # We want u such that G * [alpha, beta, gamma]^T = [t, t, t]^T
    # So [alpha, beta, gamma]^T = G^-1 * [1, 1, 1]^T * t
    # We can just solve G * x = [1, 1, 1]^T
    # Then u = normalize(x)
    # And R = arccos(u.a)
    
    ab = np.sum(a * b, axis=1)
    ac = np.sum(a * c, axis=1)
    bc = np.sum(b * c, axis=1)
    
    # Construct Gram matrix G
    # G = [[1, ab, ac],
    #      [ab, 1, bc],
    #      [ac, bc, 1]]
    
    # Solve G * x = [1, 1, 1]
    # Using Cramer's rule or direct inverse?
    # Inverse of 3x3 is fast.
    
    # det(G) = 1*(1 - bc^2) - ab*(ab - bc*ac) + ac*(ab*bc - ac)
    #         = 1 - bc^2 - ab^2 + ab^2*bc - ac^2 + ac*ab*bc
    #         = 1 - (ab^2 + bc^2 + ac^2 - 2*ab*bc*ac)
    
    det_G = 1.0 - (ab**2 + bc**2 + ac**2 - 2.0 * ab * bc * ac)
    
    # x1 = det([[1, ac], [bc, 1]]) / det_G = (1 - ac*bc) / det_G
    # x2 = det([[1, ab], [ac, bc]]) / det_G = (bc - ab*ac) / det_G
    # x3 = det([[1, ab], [1, bc]]) / det_G = (bc - ab) / det_G ... wait
    # x3 = det([[1, ab], [1, bc]])? No.
    # Replace 3rd col with [1,1,1]:
    # [[1, ab, 1], [ab, 1, 1], [ac, bc, 1]]
    # det = 1*(1 - bc) - ab*(ab - ac) + 1*(ab*bc - ac)
    #     = 1 - bc - ab^2 + ab*ac + ab*bc - ac
    
    x1 = (1.0 - ac * bc) / det_G
    x2 = (bc - ab * ac) / det_G
    x3 = (1.0 - bc - ab**2 + ab * ac + ab * bc - ac) / det_G
    
    # Wait, the RHS is [1, 1, 1].
    # Let's check x1:
    # 1*1 - ab*bc + ac*bc = 1 - bc(ab - ac)? No.
    # Cramer's rule:
    # x1 = det([[1, ac, 1], [1, bc, 1], [1, 1, 1]]) ... no, replace col 1.
    # G1 = [[1, ac, 1], [1, bc, 1], [1, 1, 1]]? No.
    # G = [[1, ab, ac], [ab, 1, bc], [ac, bc, 1]]
    # b = [1, 1, 1]
    # x1: replace col 1 with b:
    # [[1, ab, ac], [1, 1, bc], [1, bc, 1]]
    # det = 1*(1 - bc^2) - ab*(1 - bc) + ac*(bc - 1)
    #     = 1 - bc^2 - ab + ab*bc + ac*bc - ac
    #     = 1 - bc^2 - ab - ac + bc(ab + ac)
    
    # This is getting error-prone. Let's use np.linalg.solve in a loop?
    # N=50, hull has ~100 faces. 100 solves is fast.
    
    u = np.zeros_like(a)
    radii = np.zeros(len(a))
    
    for i in range(len(a)):
        ai = a[i]
        bi = b[i]
        ci = c[i]
        
        G = np.array([
            [1.0, np.dot(ai, bi), np.dot(ai, ci)],
            [np.dot(ai, bi), 1.0, np.dot(bi, ci)],
            [np.dot(ai, ci), np.dot(bi, ci), 1.0]
        ])
        
        try:
            x = np.linalg.solve(G, np.ones(3))
        except np.linalg.LinAlgError:
            # Fallback: use normal vector
            n = np.cross(ai - bi, bi - ci)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                radii[i] = 0.0
                continue
            u_i = n / norm_n
        else:
            u_i = x[0] * ai + x[1] * bi + x[2] * ci
            norm_u = np.linalg.norm(u_i)
            if norm_u < 1e-12:
                radii[i] = 0.0
                continue
            u_i = u_i / norm_u
        
        # Distance to vertex a
        dot_val = np.dot(u_i, ai)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        radii[i] = np.arccos(dot_val)
        
    max_radius = np.max(radii) if len(radii) > 0 else 0.0
    return max_radius, radii, hull

def _soft_max_radii(radii, beta):
    """Compute soft-max of radii."""
    if len(radii) == 0:
        return 0.0, np.array([])
    max_r = np.max(radii)
    if max_r < 1e-15:
        return 0.0, np.zeros_like(radii)
    shifted = radii - max_r
    weights = np.exp(beta * shifted)
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return max_r, np.zeros_like(radii)
    weights /= sum_weights
    val = max_r + np.log(sum_weights) / beta
    return val, weights

def _objective_and_grad(pts, beta):
    """
    Objective: soft-max of facet circumradii.
    Gradient: numerical or analytical?
    Analytical gradient of circumradius is complex.
    We can use finite differences or rely on L-BFGS-B with numerical grad?
    L-BFGS-B in scipy can use numerical gradient if jac is not provided, but it's slow.
    Let's provide a numerical gradient to be safe, or use 'CG' method?
    Actually, let's just use minimize with method='L-BFGS-B' and let it compute numerical gradient.
    For 150 variables, numerical gradient is 151 evaluations.
    If each evaluation is ~1ms, that's 150ms per step.
    100 steps = 15s.
    This is feasible.
    """
    max_r, radii, hull = _compute_facet_circumradii(pts)
    if max_r is None:
        return 1e10, np.zeros_like(pts)
    
    val, weights = _soft_max_radii(radii, beta)
    
    # Gradient is not easily available analytically here without deriving
    # dR/dp for each facet.
    # We will let scipy compute it numerically.
    return val, None

def _spherical_lloyd(pts, max_iter=10, tol=1e-6):
    """
    Spherical Lloyd's algorithm: move each point to the centroid of its Voronoi cell.
    Approximated by the average of the outward normals of the incident facets.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Accumulate normals for each vertex
        normals = np.zeros_like(pts)
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            # Ensure outward orientation
            if np.dot(normal, a) < 0:
                normal = -normal
            
            for idx in simplex:
                normals[idx] += normal
        
        # Move points to normalized sum of normals
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        for i in range(n):
            norm_sum = np.linalg.norm(normals[i])
            if norm_sum < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = normals[i] / norm_sum
            
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
        
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    best_pts = _fibonacci_points(N)
    best_score = _compute_facet_circumradii(best_pts)[0]
    
    if best_score is None:
        best_score = np.pi
    
    np.random.seed(42)
    
    # Initializations
    inits = [
        _fibonacci_points(N),
        _normalize(np.random.randn(N, 3)),
        _normalize(best_pts + np.random.randn(N, 3) * 0.5)
    ]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        current_pts = _normalize(init_pts.copy())
        
        # L-BFGS Optimization
        # We optimize in R^3 and project to sphere in the callback?
        # Or optimize in tangent space?
        # Simplest: optimize in R^3, normalize inside objective.
        # But L-BFGS assumes Euclidean space.
        # We can use a constraint or just normalize after each step?
        # Scipy minimize doesn't support custom projection easily.
        # We can use a wrapper that normalizes the input.
        
        def obj_wrapper(x):
            p = x.reshape(N, 3)
            p = _normalize(p)
            val, _ = _objective_and_grad(p, beta=20.0)
            return val
        
        # Start L-BFGS
        try:
            res = minimize(
                obj_wrapper,
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            current_pts = _normalize(res.x.reshape(N, 3))
        except Exception:
            pass
        
        # Refine with Lloyd's
        if time.time() < deadline - 10:
            current_pts = _spherical_lloyd(current_pts, max_iter=5)
        
        # Evaluate
        score, _, _ = _compute_facet_circumradii(current_pts)
        if score is None:
            score = np.pi
            
        if score < best_score:
            best_score = score
            best_pts = current_pts.copy()
            
            # Try a few more refinements if time permits
            if time.time() < deadline - 5:
                # Second pass of L-BFGS with higher beta
                def obj_wrapper2(x):
                    p = x.reshape(N, 3)
                    p = _normalize(p)
                    val, _ = _objective_and_grad(p, beta=50.0)
                    return val
                
                try:
                    res2 = minimize(
                        obj_wrapper2,
                        current_pts.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'ftol': 1e-9, 'gtol': 1e-6}
                    )
                    current_pts = _normalize(res2.x.reshape(N, 3))
                    current_pts = _spherical_lloyd(current_pts, max_iter=5)
                    
                    score2, _, _ = _compute_facet_circumradii(current_pts)
                    if score2 is not None and score2 < best_score:
                        best_score = score2
                        best_pts = current_pts.copy()
                except Exception:
                    pass
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
