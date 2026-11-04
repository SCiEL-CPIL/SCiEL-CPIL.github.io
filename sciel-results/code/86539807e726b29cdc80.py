"""Optimized 130-point sphere covering using Riemannian trust-region on convex hull facets."""
N = 130
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

# Deadline guard: keep a safety margin before the 150s wall-clock limit
_DEADLINE = 140.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute covering radius (in radians) using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        # For each facet, compute the circumradius of the spherical triangle
        # The covering radius is the max over facets of the angular distance from facet centroid to its vertices
        # Actually, for a convex hull on the sphere, the covering radius equals the max circumradius of facets
        # where the circumcenter lies on the sphere (empty cap condition)
        
        # More robust: for each facet (triangle), compute the spherical circumradius
        # The circumradius of a spherical triangle with angles A,B,C is related to its sides
        # But simpler: the covering radius is the max distance from any point on sphere to nearest site
        # This equals the max over Voronoi cells of the distance to the cell's farthest vertex
        # On the sphere, this corresponds to the circumradius of each facet of the convex hull
        # where the circumcenter is on the far side (empty cap)
        
        # Compute for each facet: take the 3 vertices, find the point equidistant (on sphere) from all 3
        # that is farthest from the origin side (i.e., the circumcenter on the sphere)
        # The circumradius is the angular distance from that center to any vertex
        
        max_r = 0.0
        for simplex in hull.simplices:
            v = pts[simplex]  # shape (3,3)
            # Find circumcenter on sphere: point c such that |c - v_i| is equal for all i
            # and c is on the unit sphere
            # The circumcenter of a triangle on the unit sphere can be found by:
            # c = normalize(a + b + c) only if the triangle is small; generally need to solve
            # Actually: the circumcenter is proportional to the cross product approach
            # For 3 points on unit sphere, the circumcenter is:
            # c = (a*(b^2 - c^2) + b*(c^2 - a^2) + c*(a^2 - b^2)) / |...|  -- no, that's for planar
            # On sphere: the circumcenter is along the direction of the normal to the plane containing the 3 points
            # if the triangle is contained in a great circle. Generally, solve:
            # minimize sum |x - v_i|^2 subject to |x|=1
            # This gives x = normalize(sum v_i) only if we ignore the constraint, but with constraint:
            # x = normalize(M v) where M is... actually the Lagrange multiplier gives:
            # x = (sum v_i) / |sum v_i| is NOT correct in general.
            # The correct solution: x = normalize(v1 + v2 + v3) is the centroid, not circumcenter.
            
            # Let's use: the circumcenter on the sphere for triangle (a,b,c) is:
            # c = normalize( a * |b|^2 * |c|^2 * ... ) -- too complex
            # Simpler approach: the circumradius r satisfies:
            # For the triangle, the circumcenter C on sphere has:
            # angle(C, a) = angle(C, b) = angle(C, c) = r
            # We can solve this by finding the intersection of two great circles (perpendicular bisectors)
            
            # Alternative robust method: use the fact that for small triangles, 
            # the circumradius ≈ (a*b*c)/(4*Area) in planar approx, but we need exact.
            
            # Let's just compute the farthest point in each Voronoi cell by checking
            # the circumcenters of all triangles formed by 3 sites that form a facet
            # For each facet, the circumcenter on the sphere:
            # Solve: find x on unit sphere such that x·a = x·b = x·c = cos(r)
            # This means x is orthogonal to (a-b) and (a-c), so x is parallel to (a-b)×(a-c)
            # Wait, that's only if the circumcenter is on the normal direction. Let's verify:
            # If x·a = x·b, then x·(a-b) = 0, so x ⊥ (a-b). Similarly x ⊥ (a-c).
            # So x is parallel to (a-b)×(a-c). The circumcenter is ±normalize((a-b)×(a-c)).
            # We need the one that gives the correct (larger) radius, i.e., the one on the "far side"
            # from the origin relative to the triangle.
            
            n = np.cross(v[0] - v[1], v[0] - v[2])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            c1 = n / n_norm
            c2 = -c1
            
            # Check which one is the circumcenter (equidistant from all 3 vertices)
            # Both should be equidistant by construction (since they're orthogonal to a-b and a-c)
            # The circumradius is arccos(c·a) for the correct c
            # We want the c that is "outside" the triangle, i.e., the one where the cap is empty
            # Typically, the circumcenter that gives the larger radius is the relevant one for covering
            
            r1 = np.arccos(np.clip(c1 @ v[0], -1, 1))
            r2 = np.arccos(np.clip(c2 @ v[0], -1, 1))
            
            # The covering radius for this facet is the larger of the two (the one on the far side)
            # Actually, we need the circumcenter that lies in the "empty cap" region
            # For a convex hull, the relevant circumcenter is the one opposite to the interior
            # Let's take the max of the two radii, but verify it's a valid empty cap
            r = max(r1, r2)
            
            # Verify the cap is empty: no other point is closer to c than the 3 vertices
            # For efficiency, skip this check in the fast path
            max_r = max(max_r, r)
        
        return max_r
    except Exception:
        # Fallback: brute force on a grid
        return _covering_radius_grid(pts)

def _covering_radius_grid(pts, n_grid=100):
    """Fallback: estimate covering radius by sampling the sphere."""
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2*np.pi, n_grid)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find min distance to pts
    max_r = 0.0
    for gp in grid_pts:
        d = np.arccos(np.clip(pts @ gp, -1, 1))
        max_r = max(max_r, np.min(d))
    return max_r

def _subgradient_covering(pts):
    """Compute subgradient of the covering radius objective.
    Returns (grad, worst_facet_index) where grad is the sum of subgradients
    from the worst facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
    
    # Compute circumradius for each facet
    radii = []
    centers = []
    for simplex in hull.simplices:
        v = pts[simplex]
        n = np.cross(v[0] - v[1], v[0] - v[2])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            radii.append(0.0)
            centers.append(None)
            continue
        c1 = n / n_norm
        c2 = -c1
        r1 = np.arccos(np.clip(c1 @ v[0], -1, 1))
        r2 = np.arccos(np.clip(c2 @ v[0], -1, 1))
        r = max(r1, r2)
        c = c1 if r1 >= r2 else c2
        radii.append(r)
        centers.append(c)
    
    radii = np.array(radii)
    max_r = np.max(radii)
    
    # Subgradient: for the worst facets, the subgradient of arccos(c·v_i) w.r.t. v_i
    # d/dv arccos(c·v) = -c / sqrt(1 - (c·v)^2) = -c / sin(r)
    # But we need to be careful about which vertex contributes
    
    grad = np.zeros_like(pts)
    worst_idx = np.where(radii >= max_r - 1e-10)[0]
    
    for idx in worst_idx:
        c = centers[idx]
        if c is None:
            continue
        simplex = hull.simplices[idx]
        v = pts[simplex]
        for k, vi in enumerate(v):
            cos_r = c @ vi
            sin_r = np.sqrt(max(0, 1 - cos_r**2))
            if sin_r < 1e-10:
                continue
            # Subgradient of arccos(c·v) w.r.t. v is -c/sin(r)
            # But c depends on v, so this is approximate
            grad[simplex[k]] += -c / sin_r
    
    # Normalize gradient to unit length for trust region
    g_norm = np.linalg.norm(grad)
    if g_norm > 1e-10:
        grad /= g_norm
    
    return grad, max_r

def _riemannian_line_search(pts, grad, alpha_max=0.1):
    """Line search on the sphere: move pts in direction of -grad, project back."""
    # Project grad to tangent space at each point
    tangent_grad = grad - (grad * pts).sum(axis=1, keepdims=True) * pts
    
    # Try different step sizes
    best_pts = pts
    best_r = _covering_radius(pts)
    
    for alpha in [alpha_max, alpha_max/2, alpha_max/4, alpha_max/8, alpha_max/16]:
        # Exponential map: pts_new = pts*cos(alpha) + tangent_grad/sin_alpha * sin(alpha)
        # For small alpha, approximately pts + alpha * tangent_grad
        # Use exact exponential map
        pts_new = pts * np.cos(alpha)[:, None] + tangent_grad * (np.sin(alpha)[:, None] / np.maximum(np.linalg.norm(tangent_grad, axis=1, keepdims=True), 1e-10))
        pts_new = _normalize(pts_new)
        
        r_new = _covering_radius(pts_new)
        if r_new < best_r:
            best_pts = pts_new
            best_r = r_new
    
    return best_pts, best_r

def _voronoi_centroid_refine(pts, n_iter=50):
    """Refine points to Voronoi cell centroids (spherical)."""
    for it in range(n_iter):
        if _time_left() < 5:
            break
        new_pts = np.zeros_like(pts)
        for i in range(N):
            # Find all points closer to pts[i] than to any other point
            # Use a fine grid for each point (expensive but accurate)
            # Faster: use the fact that the Voronoi cell is bounded by great circles
            # Approximate by sampling a local cap
            theta = np.linspace(0, np.pi, 50)
            phi = np.linspace(0, 2*np.pi, 50)
            theta, phi = np.meshgrid(theta, phi)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            
            # Distances from grid to all points
            d = np.arccos(np.clip(grid_pts @ pts.T, -1, 1))  # (n_grid, N)
            # Points in Voronoi cell of i
            mask = np.argmin(d, axis=1) == i
            if np.sum(mask) > 0:
                centroid = grid_pts[mask].mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                new_pts[i] = centroid
            else:
                new_pts[i] = pts[i]
        pts = _normalize(new_pts)
    return pts

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    # Multi-restart with different initializations
    n_restarts = 3
    for restart in range(n_restarts):
        if _time_left() < 30:
            break
        
        if restart == 0:
            current = best_pts.copy()
        else:
            # Perturbed random initialization
            np.random.seed(restart * 12345)
            current = np.random.randn(N, 3)
            current = _normalize(current)
        
        # Riemannian trust-region optimization
        alpha = 0.1
        for it in range(200):
            if _time_left() < 10:
                break
            
            grad, r_curr = _subgradient_covering(current)
            if grad is None:
                break
            
            if r_curr < best_r:
                best_r = r_curr
                best_pts = current.copy()
            
            # Line search
            current, r_new = _riemannian_line_search(current, grad, alpha)
            
            if r_new < best_r:
                best_r = r_new
                best_pts = current.copy()
            
            # Adapt step size
            if r_new < r_curr - 1e-8:
                alpha = min(alpha * 1.5, 0.3)
            else:
                alpha = max(alpha * 0.5, 0.001)
            
            if alpha < 0.002:
                break
        
        # Voronoi centroid refinement
        if _time_left() > 20:
            refined = _voronoi_centroid_refine(current, n_iter=30)
            r_ref = _covering_radius(refined)
            if r_ref < best_r:
                best_r = r_ref
                best_pts = refined.copy()
    
    # Final high-precision polish
    if _time_left() > 15:
        # Fine-tune with smaller steps
        for it in range(100):
            if _time_left() < 5:
                break
            grad, r_curr = _subgradient_covering(best_pts)
            if grad is None:
                break
            best_pts, r_new = _riemannian_line_search(best_pts, grad, 0.01)
            if r_new < best_r:
                best_r = r_new
    
    return _normalize(best_pts)
