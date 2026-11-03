"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: grid
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter of spherical triangle a,b,c
        # Vector v is normal to the plane defined by a, b, c? No, a-b and b-c are in the plane.
        # Cross product of (a-b) and (b-c) gives a vector normal to the plane containing a,b,c.
        # This vector is also the direction of the circumcenter of the spherical triangle?
        # Yes, the circumcenter of a spherical triangle is the projection of the origin onto the plane 
        # defined by the three points, normalized. Wait.
        # The circumcenter of a spherical triangle is the point equidistant from a, b, c.
        # It lies on the normal to the plane through a,b,c passing through the origin?
        # Actually, the circumcenter $C$ satisfies $C \cdot a = C \cdot b = C \cdot c = \cos(R)$.
        # The vector $v = (a-b) \times (b-c)$ is normal to the plane containing a,b,c.
        # Since a,b,c are on the unit sphere, the plane does not pass through the origin (unless degenerate).
        # The line through the origin and the circumcenter is parallel to the normal of the plane?
        # Yes. The circumcenter is $\pm v / ||v||$.
        
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            # Degenerate triangle, use max edge length / 2 as approx
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / norm_v
        
        # Determine correct sign: the circumcenter should be on the same side of the face 
        # as the interior of the polyhedron? 
        # For a convex hull, the facet normal should point outward.
        # The circumcenter of the facet (on the sphere) is the point furthest from the origin 
        # in the direction of the facet normal? No.
        # The circumcenter is the point on the sphere equidistant to a,b,c.
        # There are two such points: u and -u.
        # The relevant one is the one that is the "center" of the Voronoi cell facet.
        # This is the one that is on the same side as the rest of the sphere?
        # Actually, for a convex hull of points on the sphere, the Voronoi cell of a point 
        # is the intersection of hemispheres. The vertex of the Voronoi cell (circumcenter of facet)
        # is the point $C$ such that $C \cdot a = C \cdot b = C \cdot c$ and $C$ is the local maximum 
        # of the distance to the nearest point? No, it's the local minimum of distance to the *nearest* point?
        # The covering radius is the max distance to the *nearest* point.
        # The worst point in a Voronoi cell is the vertex of the cell (circumcenter of the facet).
        # So we need the circumcenter that is *inside* the Voronoi cell.
        
        # Let's check the normal of the facet.
        n_face = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n_face)
        if norm_n < 1e-12:
            continue
        n_face = n_face / norm_n
        
        # Ensure n_face points outward (away from origin)
        # For a convex hull, the centroid of the facet is on the surface.
        # The vector from origin to centroid is roughly parallel to n_face.
        # Actually, for a unit sphere, the normal to the plane abc is parallel to the circumcenter.
        # Let's check the dot product with the centroid of the triangle.
        centroid = (a + b + c) / 3.0
        if np.dot(n_face, centroid) < 0:
            n_face = -n_face
            
        # The circumcenter u or -u should have the same sign of dot product with n_face?
        # The circumcenter is the point on the sphere.
        # If u is the circumcenter, then u is orthogonal to the plane? No, u is normal to the plane.
        # So u is parallel to n_face.
        # If np.dot(u, n_face) > 0, then u is the outward circumcenter.
        # The Voronoi vertex is the one that is "outside" the triangle in terms of distance?
        # Actually, the Voronoi cell is the set of points closer to a than any other.
        # The vertex is the intersection of the bisector planes.
        # The bisector plane of a and b is $x \cdot (a-b) = 0$.
        # The normal to this plane is $a-b$.
        # The vertex $C$ satisfies $C \cdot (a-b) = 0$, etc.
        # So $C$ is orthogonal to $a-b$ and $b-c$. Thus $C$ is parallel to $(a-b) \times (b-c) = v$.
        # So $C = \pm u$.
        # Which one? The one that is further from the origin? Both are on the sphere.
        # The one that is the "center" of the cap defined by a,b,c.
        # If the triangle is small, the circumcenter is close to the centroid.
        # Centroid is roughly in direction $a+b+c$.
        # So we should pick the sign of u that is consistent with $a+b+c$.
        
        if np.dot(u, a + b + c) < 0:
            circum_center = -u
        else:
            circum_center = u
            
        # Calculate distance from circum_center to a (should be same for b, c)
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        # d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        # d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        # cr = max(d_a, d_b, d_c) # They should be equal
        
        max_cr = max(max_cr, d_a)
        
    return max_cr

def _covering_radius_grid(pts, n_grid=200):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _spherical_lloyd(pts, max_iter=100, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        incident_facets = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            centroid = np.sum(incident_facets[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
                
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
            
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _soft_max_covering_radius(pts, beta=100.0):
    """
    Compute soft-max covering radius and its gradient w.r.t. point positions.
    Returns (soft_max_cr, gradient_array).
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0, np.zeros((n, 3))
    
    crs = []
    ccs = []
    simplices = []
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v = np.cross(a - b, b - c)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            continue
        u = v / norm_v
        
        # Determine sign
        if np.dot(u, a + b + c) < 0:
            cc = -u
        else:
            cc = u
            
        d_a = np.arccos(np.clip(np.dot(cc, a), -1, 1))
        crs.append(d_a)
        ccs.append(cc)
        simplices.append(simplex)
    
    if not crs:
        return 1.0, np.zeros((n, 3))
    
    crs = np.array(crs)
    max_cr = np.max(crs)
    
    if max_cr < 1e-10:
        return max_cr, np.zeros((n, 3))
    
    # Soft-max
    scaled = beta * (crs - max_cr)
    log_sum_exp = max_cr * beta + np.log(np.sum(np.exp(scaled)))
    soft_max_cr = log_sum_exp / beta
    
    weights = np.exp(scaled) / np.sum(np.exp(scaled))
    
    grad = np.zeros((n, 3))
    for simplex, cc, w in zip(simplices, ccs, weights):
        if w < 1e-10:
            continue
        for idx in simplex:
            dot_val = np.dot(cc, pts[idx])
            dot_val = np.clip(dot_val, -1 + 1e-15, 1 - 1e-15)
            sin_d = np.sqrt(1 - dot_val * dot_val)
            if sin_d < 1e-15:
                continue
            vec = cc / sin_d
            vec_tan = vec - np.dot(vec, pts[idx]) * pts[idx]
            grad[idx] += w * vec_tan
    
    return soft_max_cr, grad

def _polish(pts, deadline, tol=1e-12):
    """
    High-precision polish using L-BFGS-B on a smooth soft-max covering radius objective.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    try:
        def objective_and_grad(x):
            p = x.reshape(n, 3)
            p = _normalize(p)
            val, grad = _soft_max_covering_radius(p, beta=100.0)
            # Project gradient to tangent space
            grad_proj = np.zeros_like(grad)
            for i in range(n):
                grad_proj[i] = grad[i] - np.dot(grad[i], p[i]) * p[i]
            return val, grad_proj.ravel()
        
        x0 = pts.ravel()
        bounds = [(-1.5, 1.5)] * (n * 3)
        
        # Use L-BFGS-B
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 100,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 50
            }
        )
        
        if result.fun < best_cr:
            new_pts = _normalize(result.x.reshape(n, 3))
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_pts = new_pts
                best_cr = new_cr
                
    except Exception:
        pass
    
    # Additional local polish with small steps if time allows
    if time.time() < deadline - 2:
        pts = best_pts
        step_size = 0.001
        beta = 200.0
        
        for it in range(100):
            if time.time() > deadline - 1:
                break
                
            val, grad = _soft_max_covering_radius(pts, beta=beta)
            
            if val < 1e-12:
                break
            
            grad_proj = np.zeros_like(grad)
            for i in range(n):
                grad_proj[i] = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
            
            # Normalize gradient
            grad_norms = np.linalg.norm(grad_proj, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-15] = 1.0
            grad_proj = grad_proj / grad_norms
            
            max_move = 0.0
            for i in range(n):
                if np.linalg.norm(grad_proj[i]) > 1e-10:
                    pts[i] = pts[i] + step_size * grad_proj[i]
                    move = np.linalg.norm(grad_proj[i]) * step_size
                    max_move = max(max_move, move)
            
            pts = _normalize(pts)
            
            if max_move < tol:
                break
            
            step_size *= 0.99
            
        final_cr = _covering_radius(pts)
        if final_cr < best_cr:
            best_pts = pts
            best_cr = final_cr
    
    return _normalize(best_pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pts = _fibonacci_points(N)
    best_cr = _covering_radius(best_pts)
    
    np.random.seed(42)
    
    restart_count = 0
    while time.time() < deadline - 20:  # Leave 20s for final polish
        restart_count += 1
        
        if restart_count == 1:
            pts = _fibonacci_points(N)
        else:
            # Perturb best solution
            noise_scale = 0.1 * (best_cr / 0.3)
            if noise_scale > 0.3:
                noise_scale = 0.3
            if noise_scale < 0.01:
                noise_scale = 0.01
            noise = noise_scale * np.random.randn(N, 3)
            pts = best_pts + noise
            pts = _normalize(pts)
            
        # 1. Lloyd's relaxation
        pts = _spherical_lloyd(pts, max_iter=50)
        
        # 2. High-precision polish
        if time.time() < deadline - 10:
            pts = _polish(pts, deadline=deadline - 5, tol=1e-12)
        
        if time.time() > deadline - 10:
            break
            
        cr = _covering_radius(pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    # Final polish on best solution
    if time.time() < deadline - 5:
        best_pts = _polish(best_pts, deadline=deadline - 1, tol=1e-13)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
