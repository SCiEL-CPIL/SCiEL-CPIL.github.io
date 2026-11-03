"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = points / norms

    # Build convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2  # fallback: worst case
    
    # For each facet, find the circumcenter and its distance to the origin
    # The covering radius is the max over facets of the angular distance from origin to facet circumcenter
    # Actually, the covering radius is the max over facets of the angle whose sine is |circumcenter| / (something)
    # More precisely: for a facet with vertices v1, v2, ..., vk, the farthest point on the sphere 
    # from all vertices is the point on the sphere closest to the facet's circumcenter (projected).
    # The angular distance from the origin to the facet plane is what matters.
    # 
    # For a facet, the covering radius contribution is the angular distance from the origin 
    # to the farthest point on that facet's circumscribed circle on the sphere.
    # 
    # A simpler approach: for each facet, compute the circumcenter of the facet in 3D,
    # then the angular distance from the origin to that circumcenter is the covering radius for that facet.
    # But we need to be careful: the circumcenter might be outside the facet.
    #
    # Alternative: the covering radius is the maximum, over all facets, of the angle theta 
    # where cos(theta) = (distance from origin to facet plane) / (radius of circumcircle of facet on sphere)
    # Actually, let me think again.
    #
    # The covering radius is the smallest r such that caps of angular radius r centered at the points cover the sphere.
    # The worst point is at the circumcenter of a facet of the convex hull.
    # For a facet with vertices p1, ..., pk, the circumcenter c is the point equidistant from all pi.
    # The angular distance from origin to c is the covering radius for that facet.
    # But c might not be the actual worst point if it's outside the facet.
    #
    # Let's use a direct approach: for each facet, find the point on the sphere that maximizes 
    # the minimum angular distance to the facet vertices. This is the circumcenter if it lies 
    # within the facet, otherwise it's on the boundary.
    #
    # For simplicity and speed, let's compute the circumcenter of each facet and check if it's 
    # within the facet. If not, we need to check edges and vertices.
    #
    # Actually, a cleaner way: the covering radius is the max over facets of the circumradius 
    # of the spherical polygon formed by the facet. For a spherical triangle, the circumradius 
    # can be computed from the side lengths.
    #
    # Let me use a simpler numerical approach: for each facet, sample points and find the max min-distance.
    # But that's slow. Let me use the circumcenter approach.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        # Compute circumcenter of these vertices
        # The circumcenter c satisfies |c - vi| = |c - vj| for all i, j
        # This means c is in the direction of the normal to the plane through the vertices
        # (if they're on a sphere centered at origin, the circumcenter is along the normal)
        
        # For a simplex of k+1 vertices, the circumcenter is the point equidistant from all vertices.
        # Since all vertices are on the unit sphere, the circumcenter lies on the line through 
        # the origin and the centroid of the facet (for a planar facet).
        
        # Compute the normal to the facet
        if len(verts) >= 3:
            # Use the first three vertices to define the plane
            v0, v1, v2 = verts[0], verts[1], verts[2]
            n = np.cross(v1 - v0, v2 - v0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # The circumcenter is at t*n where t is chosen so that |t*n - v0| = 1 (on unit sphere)
            # |t*n - v0|^2 = 1
            # t^2 - 2t*(n·v0) + |v0|^2 = 1
            # t^2 - 2t*(n·v0) + 1 = 1
            # t^2 - 2t*(n·v0) = 0
            # t = 2*(n·v0)
            dot = np.dot(n, v0)
            t = 2.0 * dot
            c = t * n
            
            # Check if c is on the unit sphere
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-12:
                continue
            c = c / c_norm
            
            # Check if c is within the facet (i.e., the angular distance from c to each vertex is 
            # less than or equal to the covering radius, and c is in the correct hemisphere)
            # For a valid circumcenter, the dot product of (c - vi) with the facet normal should be 
            # consistent, and c should be in the convex hull of the vertices on the sphere.
            
            # A simpler check: the angular distance from c to each vertex should be the same (within tolerance)
            dists = np.arccos(np.clip(np.dot(c, verts), -1.0, 1.0))
            if np.max(np.abs(dists - dists[0])) > 0.1:  # 0.1 rad tolerance
                # Circumcenter is not valid for this facet, skip
                continue
            
            # The covering radius for this facet is the angular distance from origin to c
            # which is arccos(0) = pi/2? No, that's not right.
            # The angular distance from a point y on the sphere to the nearest vertex is what we want.
            # If c is the circumcenter, then the distance from c to each vertex is the covering radius.
            max_radius = max(max_radius, np.max(dists))
        else:
            # Degenerate case
            continue
    
    return np.degrees(max_radius)


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_covering(points, max_iter=10000, tol=1e-8):
    """Optimize the covering radius using gradient descent on a smooth surrogate."""
    pts = points.copy()
    n = len(pts)
    
    # Use a smooth surrogate: sum of exp(k * min_dist) or similar
    # Actually, let's use a direct approach: minimize the max over facets
    # We'll use a log-sum-exp approximation of the max
    
    k = 50.0  # sharpness parameter for log-sum-exp
    
    def objective_and_grad(x):
        pts = x.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Compute pairwise distances
        # For each point, find the nearest neighbor distance
        # But we want the covering radius, which is about the convex hull facets
        
        # Approximate: for each point, the local covering radius is related to the 
        # distance to its nearest neighbors. A good surrogate is the max over points 
        # of the distance to the 3rd nearest neighbor (or similar).
        
        # Compute all pairwise dot products
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # For each point, sort distances to other points
        np.fill_diagonal(dists, np.inf)
        sorted_dists = np.sort(dists, axis=1)
        
        # Use the 3rd nearest neighbor distance as a proxy for local covering radius
        # This is a rough surrogate
        local_radii = sorted_dists[:, 2]  # 3rd nearest
        
        # Log-sum-exp approximation of max
        max_local = np.max(local_radii)
        # Smooth max
        smooth_max = max_local + np.log(np.sum(np.exp(k * (local_radii - max_local)))) / k
        
        # Gradient: approximate by finite differences or analytic
        # For simplicity, use finite differences with a small step
        eps = 1e-5
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(3):
                x_plus = x.copy()
                x_plus[i * 3 + j] += eps
                pts_plus = x_plus.reshape(-1, 3)
                norms_plus = np.linalg.norm(pts_plus, axis=1, keepdims=True)
                norms_plus[norms_plus == 0] = 1.0
                pts_plus = pts_plus / norms_plus
                dots_plus = pts_plus @ pts_plus.T
                dots_plus = np.clip(dots_plus, -1.0, 1.0)
                dists_plus = np.arccos(dots_plus)
                np.fill_diagonal(dists_plus, np.inf)
                sorted_dists_plus = np.sort(dists_plus, axis=1)
                local_radii_plus = sorted_dists_plus[:, 2]
                max_local_plus = np.max(local_radii_plus)
                smooth_max_plus = max_local_plus + np.log(np.sum(np.exp(k * (local_radii_plus - max_local_plus)))) / k
                
                x_minus = x.copy()
                x_minus[i * 3 + j] -= eps
                pts_minus = x_minus.reshape(-1, 3)
                norms_minus = np.linalg.norm(pts_minus, axis=1, keepdims=True)
                norms_minus[norms_minus == 0] = 1.0
                pts_minus = pts_minus / norms_minus
                dots_minus = pts_minus @ pts_minus.T
                dots_minus = np.clip(dots_minus, -1.0, 1.0)
                dists_minus = np.arccos(dots_minus)
                np.fill_diagonal(dists_minus, np.inf)
                sorted_dists_minus = np.sort(dists_minus, axis=1)
                local_radii_minus = sorted_dists_minus[:, 2]
                max_local_minus = np.max(local_radii_minus)
                smooth_max_minus = max_local_minus + np.log(np.sum(np.exp(k * (local_radii_minus - max_local_minus)))) / k
                
                grad[i * 3 + j] = (smooth_max_plus - smooth_max_minus) / (2 * eps)
        
        return smooth_max, grad
    
    # Use L-BFGS-B
    try:
        result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B', 
                          jac=True, options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6})
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return pts


def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Try optimization with time budget
    remaining = deadline - time.time()
    if remaining > 10.0:
        # Use L-BFGS with a time limit
        try:
            optimized = _optimize_covering(points, max_iter=5000, tol=1e-7)
            # Normalize
            norms = np.linalg.norm(optimized, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            optimized = optimized / norms
            
            radius = _covering_radius(optimized)
            if radius < best_radius:
                best_points = optimized
                best_radius = radius
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_points
    
    # Try a few random restarts if time allows
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        
        # Random perturbation
        perturbed = best_points + 0.01 * rng.randn(N, 3)
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        perturbed = perturbed / norms
        
        try:
            optimized = _optimize_covering(perturbed, max_iter=2000, tol=1e-6)
            norms = np.linalg.norm(optimized, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            optimized = optimized / norms
            
            radius = _covering_radius(optimized)
            if radius < best_radius:
                best_points = optimized
                best_radius = radius
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
