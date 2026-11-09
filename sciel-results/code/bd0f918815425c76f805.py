"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            pts = points[simplex]
            # Centroid of the facet
            c = np.mean(pts, axis=0)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c = c / c_norm
            # Distance from origin to plane through facet
            # The circumcenter of the facet projected onto the sphere
            # For a triangle on unit sphere, the circumcenter direction is the normalized cross product approach
            # Simpler: find point on sphere furthest from all three vertices
            # The covering radius for this facet is the angular distance from the facet center (on sphere) to the farthest vertex
            # Actually, the covering radius is the max over sphere of min angular dist to any point.
            # The worst point for a facet is the circumcenter of that facet (if it's on the sphere side)
            # For a spherical triangle, the circumcenter is the direction of the normalized vector sum if acute
            # Let's use: the point on the unit sphere that maximizes min distance to the three vertices
            # This is the spherical circumcenter. For a triangle with vertices a,b,c on unit sphere,
            # the circumcenter direction is proportional to (a x b) x (b x c) ... 
            # Simpler robust method: the circumcenter of the planar triangle, normalized
            # But we need the spherical circumcenter.
            # Use the fact that the circumcenter is equidistant from all three vertices.
            # Solve: |x - a|^2 = |x - b|^2 = |x - c|^2, |x| = 1
            # This gives x = (a + b + c) normalized if the triangle is acute, but not always.
            # More robust: x is proportional to cross(a-b, a-c) for the normal, but that's the plane normal.
            # Actually the spherical circumcenter is the intersection of the sphere with the line through the origin
            # and the circumcenter of the planar triangle... no that's not right either.
            # Let me use a different approach: for each facet, the covering radius contribution is the
            # angular distance from the "center" of the empty cap to the nearest vertex.
            # The center of the empty cap for a facet is the point on the sphere that is equidistant to all
            # vertices of the facet and lies on the same side.
            # For a triangle a,b,c on unit sphere, the circumcenter x satisfies:
            # x·a = x·b = x·c, |x|=1
            # From x·a = x·b: x·(a-b) = 0
            # From x·a = x·c: x·(a-c) = 0
            # So x is parallel to (a-b) x (a-c), which is the normal to the plane.
            # Wait, that means x is the normalized normal to the facet plane!
            # Let's verify: if x = n/|n| where n = (a-b)x(a-c), then x·(a-b)=0 and x·(a-c)=0,
            # so x·a = x·b and x·a = x·c. Yes!
            # But we need x to be on the correct side (outside the convex hull, i.e., the cap side).
            n = np.cross(pts[0] - pts[1], pts[0] - pts[2])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            x = n / n_norm
            # Make sure x points outward (away from origin relative to facet center)
            if np.dot(x, c) < 0:
                x = -x
            # The covering radius for this facet is the angular distance from x to any vertex
            # (they should all be equal)
            cos_dist = np.dot(x, pts[0])
            cos_dist = np.clip(cos_dist, -1.0, 1.0)
            ang = np.arccos(cos_dist)
            max_r = max(max_r, ang)
        return max_r
    except Exception:
        return np.pi

def _repulsion_energy(points):
    """Sum of inverse squared distances to nearest neighbor (repulsion)."""
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq, axis=1)
    return np.sum(1.0 / min_dist_sq)

def _optimize(points, max_iters=200, tol=1e-8):
    """Local optimization using gradient descent on a combined objective."""
    pts = points.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    lr = 0.01
    for it in range(max_iters):
        # Compute nearest neighbor distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Gradient of repulsion energy: for each point i, gradient w.r.t. x_i
        # E = sum_j 1/d_j^2 where d_j is min distance for point j
        # dE/dx_i = -2 * sum over j where i is nearest to j of (1/d_j^4) * (x_j - x_i)
        # This is complex; use numerical gradient or a simpler approach.
        
        # Alternative: use the force-directed approach
        # Each point feels repulsion from its nearest neighbor
        forces = np.zeros_like(pts)
        for i in range(len(pts)):
            # Find nearest neighbor
            d = dist_sq[i]
            j = np.argmin(d)
            if d[j] < 1e-12:
                continue
            # Repulsive force away from nearest neighbor
            f = (pts[i] - pts[j]) / d[j] ** 1.5
            forces[i] += f
        
        # Project forces onto tangent plane of sphere
        for i in range(len(pts)):
            tangent = forces[i] - np.dot(forces[i], pts[i]) * pts[i]
            pts[i] += lr * tangent
            # Renormalize
            norm = np.linalg.norm(pts[i])
            if norm > 1e-10:
                pts[i] /= norm
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
        
        if it % 10 == 0 and lr > 0.001:
            lr *= 0.95
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_cr = np.pi
    
    # Multiple restarts with different initializations
    rng = np.random.default_rng(42)
    
    # Restart 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Restart 2: Random
    pts1 = rng.standard_normal((N, 3))
    pts1 /= np.linalg.norm(pts1, axis=1, keepdims=True)
    
    # Restart 3: Random with slight perturbation of Fibonacci
    pts2 = pts0 + 0.1 * rng.standard_normal((N, 3))
    pts2 /= np.linalg.norm(pts2, axis=1, keepdims=True)
    
    # Restart 4: Different golden angle
    i4 = np.arange(N) + 0.5
    z4 = 1.0 - 2.0 * i4 / N
    phi4 = np.pi * (1.0 + np.sqrt(5.0)) * i4 * 1.618
    r4 = np.sqrt(1.0 - z4 * z4)
    pts3 = np.column_stack([r4 * np.cos(phi4), r4 * np.sin(phi4), z4])
    
    initializations = [pts0, pts1, pts2, pts3]
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        # Run local optimization
        opt_pts, cr = _optimize(init_pts, max_iters=300)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = opt_pts.copy()
        
        # Try a few more iterations with smaller learning rate
        if time.time() < deadline - 20:
            # Fine-tune
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                fine_pts, fine_cr = _optimize(best_pts, max_iters=100)
                if fine_cr < best_cr:
                    best_cr = fine_cr
                    best_pts = fine_pts.copy()
    
    if best_pts is None:
        best_pts = pts0
    
    return best_pts
# EVOLVE-BLOCK-END
