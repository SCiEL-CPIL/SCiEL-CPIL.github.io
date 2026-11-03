"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint


def _normalize(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum, over all points on the sphere,
    of the angular distance to the nearest point in the configuration.
    
    This equals the largest circumradius of the facets of the convex hull
    (for the dual problem on the sphere). We compute it by checking
    each facet of the convex hull.
    
    For efficiency, we use the fact that the covering radius is the
    maximum over all points y on the sphere of min_i angle(y, p_i).
    This can be found by: for each point p_i, the Voronoi cell on the
    sphere, and the farthest point in that cell.
    
    A simpler approach: the covering radius is the largest value such that
    there exists a point on the sphere at distance >= that value from all
    p_i. Equivalently, it's the maximum circumradius of the convex hull
    facets.
    
    We'll compute it via: for each pair/triple of points that form a facet
    of the convex hull, compute the circumradius. But finding convex hull
    facets is expensive.
    
    Alternative: Use the fact that the covering radius r satisfies:
    max_y min_i arccos(y . p_i) = r
    
    We can find this by: the worst point y* is either a vertex of the 
    Voronoi diagram (intersection of 3 great circles) or on an edge.
    
    For N=50, a practical approach: 
    1. Compute all pairwise distances
    2. The covering radius is bounded by half the max nearest-neighbor distance
    3. More precisely, use scipy's ConvexHull on the 3D points, then for each
       facet compute the circumradius.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial import ConvexHullError
    
    try:
        hull = ConvexHull(pts)
    except ConvexHullError:
        # Degenerate case
        return np.pi
    
    max_circum = 0.0
    
    for simplex in hull.simplices:
        # simplex has 3 points (triangular facet in 3D)
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute circumradius of the spherical triangle
        # The circumradius is the angular distance from the circumcenter
        # to each vertex.
        
        # Method: find point x on sphere equidistant from a, b, c
        # x . a = x . b = x . c = cos(r)
        # This means x is in the null space of [a; b; c] after centering...
        # Actually, x . a = x . b = x . c means:
        # (a - b) . x = 0 and (b - c) . x = 0
        # So x is parallel to (a-b) x (b-c) or the cross product approach
        
        # Better: x is equidistant from a, b, c on the sphere.
        # The circumcenter of a spherical triangle can be found as:
        # x = normalize(a * sin(2*A) + b * sin(2*B) + c * sin(2*C))
        # where A, B, C are the angles of the spherical triangle.
        
        # Simpler numerical approach:
        # Solve for x on unit sphere with x.a = x.b = x.c
        # This is: x is in the intersection of the unit sphere with the
        # plane (a-b).x = 0 and (b-c).x = 0
        # So x is parallel to (a-b) x (b-c), or its negative.
        
        v1 = a - b
        v2 = b - c
        x_candidate = np.cross(v1, v2)
        x_norm = np.linalg.norm(x_candidate)
        
        if x_norm < 1e-12:
            # Degenerate: a, b, c are collinear (or two coincide)
            # In this case, the "circumcenter" is ambiguous
            # The covering radius contribution is half the max distance
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ac = np.arccos(np.clip(np.dot(a, c), -1, 1))
            max_d = max(d_ab, d_bc, d_ac)
            max_circum = max(max_circum, max_d / 2.0)
            continue
        
        x = x_candidate / x_norm
        
        # Check if x is on the same side as the facet normal
        # The facet normal points outward from the convex hull
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter should be on the opposite side of the facet
        # from the origin (since the hull contains the origin for a good covering)
        # Actually, for the covering radius, we want the farthest point in
        # the Voronoi cell, which is the circumcenter of the facet if it
        # lies on the correct side.
        
        # Compute the angular distance from x to a (should be same to b and c)
        cos_r = np.dot(x, a)
        cos_r = np.clip(cos_r, -1, 1)
        r = np.arccos(cos_r)
        
        # Also check the antipodal point
        x_neg = -x
        cos_r_neg = np.dot(x_neg, a)
        cos_r_neg = np.clip(cos_r_neg, -1, 1)
        r_neg = np.arccos(cos_r_neg)
        
        # The relevant circumradius is the one where the center is in the
        # Voronoi cell. For a proper covering, the circumcenter should be
        # "above" the facet (away from the center of the sphere/hull).
        
        # Use the one that gives the larger distance (more conservative)
        # Actually, we need the one that's in the Voronoi cell of the facet.
        # The Voronoi cell for a facet is on the side of the facet away from
        # the other points. Since the hull contains the origin (for a good
        # covering), the relevant circumcenter is the one farther from the
        # origin, which is just x or -x depending on orientation.
        
        # The covering radius is the max over all facets of the circumradius
        # where the circumcenter is in the Voronoi cell.
        # For a convex hull containing the origin, the circumcenter of each
        # facet that lies outside the hull (i.e., on the side of the facet
        # away from the origin) gives a valid covering radius candidate.
        
        # Check which side x is on relative to the facet
        # Facet plane: n . (y - a) = 0, i.e., n . y = n . a
        # Origin is at 0, so n . 0 = 0. If n . a > 0, origin is on the side
        # where n . y < n . a.
        
        side_origin = np.dot(n, np.zeros(3)) - np.dot(n, a)  # = -n.a
        side_x = np.dot(n, x) - np.dot(n, a)
        
        # If x is on the opposite side from the origin, it's a valid circumcenter
        if side_x * side_origin < 0:
            max_circum = max(max_circum, r)
        else:
            # Try the antipodal point
            side_x_neg = np.dot(n, x_neg) - np.dot(n, a)
            if side_x_neg * side_origin < 0:
                max_circum = max(max_circum, r_neg)
            else:
                # Both are on the same side as origin or degenerate
                # Use the larger one as a safe estimate
                max_circum = max(max_circum, max(r, r_neg))
    
    return max_circum


def _soft_covering_radius(pts, beta):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    We want to minimize max_y min_i angle(y, p_i).
    
    A common approach: minimize a smooth approximation of the max.
    The covering radius can be approximated by considering the Delaunay
    triangulation and the circumradii.
    
    Alternative smooth objective: 
    For each point y, d(y) = min_i arccos(y . p_i)
    We want to minimize max_y d(y).
    
    A smooth surrogate: use the fact that the worst-case point is a 
    circumcenter of some Delaunay triangle. We can use a soft-max over
    the circumradii of all triples.
    
    But for N=50, there are C(50,3) = 19600 triples, which is manageable.
    For each triple, compute the circumradius, then take a soft-max.
    """
    # Compute circumradii for all triples
    # This is expensive for N=50 (19600 triples), but doable.
    
    # Actually, let's use a different approach: 
    # The covering radius is the smallest r such that the union of 
    # spherical caps of radius r around each p_i covers the sphere.
    # Equivalently, it's the largest circumradius of the Voronoi cells.
    
    # For a smooth objective, we can use:
    # f(pts) = log-sum-exp over all triples of (beta * circumradius(triple))
    # This approximates max_triple circumradius(triple) as beta -> infinity.
    
    # But computing circumradii for all triples is O(N^3) which is ~20k for N=50.
    # That's fine.
    
    idx = np.arange(N)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    k_flat = k_idx.ravel()
    
    # Keep only i < j < k
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    a = pts[i_sel]
    b = pts[j_sel]
    c = pts[k_sel]
    
    # Compute circumradius for each triple
    # x is parallel to (a-b) x (b-c)
    v1 = a - b
    v2 = b - c
    x = np.cross(v1, v2)
    x_norms = np.linalg.norm(x, axis=1)
    
    circum_radii = np.zeros(len(i_sel))
    
    valid = x_norms > 1e-12
    if np.any(valid):
        x_valid = x[valid] / x_norms[valid, None]
        # Circumradius = arccos(x . a)
        cos_r = np.sum(x_valid * a[valid], axis=1)
        cos_r = np.clip(cos_r, -1, 1)
        circum_radii[valid] = np.arccos(cos_r)
    
    # Also consider the antipodal point for each triple
    if np.any(valid):
        x_valid_neg = -x_valid
        cos_r_neg = np.sum(x_valid_neg * a[valid], axis=1)
        cos_r_neg = np.clip(cos_r_neg, -1, 1)
        circum_radii_neg = np.arccos(cos_r_neg)
        # Take the max of the two (conservative)
        circum_radii[valid] = np.maximum(circum_radii[valid], circum_radii_neg)
    
    # For invalid triples (collinear), use half the max pairwise distance
    if np.any(~valid):
        d_ab = np.arccos(np.clip(np.sum(a[~valid] * b[~valid], axis=1), -1, 1))
        d_bc = np.arccos(np.clip(np.sum(b[~valid] * c[~valid], axis=1), -1, 1))
        d_ac = np.arccos(np.clip(np.sum(a[~valid] * c[~valid], axis=1), -1, 1))
        circum_radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ac) / 2.0
    
    # Soft-max
    beta_scaled = beta * circum_radii
    max_val = np.max(beta_scaled)
    if max_val > 700:
        return max_val / beta
    log_sum = max_val + np.log(np.sum(np.exp(beta_scaled - max_val)))
    return log_sum / beta


def _soft_covering_radius_grad(pts, beta):
    """
    Compute the gradient of the soft covering radius with respect to pts.
    This is complex; we'll use numerical gradients or a simplified version.
    """
    # For simplicity, use finite differences for the gradient
    # This is slow but correct.
    eps = 1e-7
    grad = np.zeros_like(pts)
    
    for i in range(N):
        for j in range(3):
            pts_plus = pts.copy()
            pts_plus[i, j] += eps
            pts_plus = _normalize(pts_plus)
            f_plus = _soft_covering_radius(pts_plus, beta)
            
            pts_minus = pts.copy()
            pts_minus[i, j] -= eps
            pts_minus = _normalize(pts_minus)
            f_minus = _soft_covering_radius(pts_minus, beta)
            
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_pts = _normalize(best_pts)
    best_radius = _covering_radius(best_pts)
    
    # Multi-beta L-BFGS cascade
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        
        current_pts = best_pts.copy()
        
        # Try a few random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 15.0:
                break
            
            if restart == 0:
                pts = current_pts.copy()
            else:
                # Perturb
                noise = np.random.randn(N, 3) * 0.1
                pts = _normalize(current_pts + noise)
            
            # Optimize
            try:
                result = minimize(
                    _soft_covering_radius,
                    pts,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
                    args=(beta,)
                )
                
                if result.success:
                    opt_pts = _normalize(result.x)
                    opt_radius = _covering_radius(opt_pts)
                    
                    if opt_radius < best_radius:
                        best_pts = opt_pts
                        best_radius = opt_radius
                        current_pts = opt_pts.copy()
                        
            except Exception:
                pass
        
        # Update best
        if time.time() > deadline - 10.0:
            break
    
    # Final polish: use the actual covering radius with a local search
    if time.time() < deadline - 5.0:
        # Try a few more iterations with the true objective
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Random perturb and accept if better
            improved = False
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                idx = np.random.randint(N)
                noise = np.random.randn(3) * 0.01
                new_pts = best_pts.copy()
                new_pts[idx] = _normalize(new_pts[idx] + noise)[0]
                new_pts = _normalize(new_pts)
                
                new_radius = _covering_radius(new_pts)
                if new_radius < best_radius - 1e-10:
                    best_pts = new_pts
                    best_radius = new_radius
                    improved = True
            
            if not improved:
                break
    
    return best_pts
# EVOLVE-BLOCK-END
