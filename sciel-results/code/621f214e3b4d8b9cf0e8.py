"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 50
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius: max over all points y on sphere of angular distance
    to nearest point in pts. This equals the maximum circumradius of the spherical
    facets (Voronoi cells) of the convex hull.
    
    For points on the unit sphere, the covering radius is the maximum distance from
    any point on the sphere to the nearest site. This can be found by computing the
    circumcenters of the spherical triangles (Voronoi cells) and checking if they
    are valid (inside the cell), or by using the convex hull approach.
    
    A robust approach: the covering radius is the maximum over all points on the sphere
    of the distance to the nearest site. This is equivalent to finding the largest empty
    spherical cap. The center of such a cap is either at a site or at a point equidistant
    from 3+ sites (a Voronoi vertex).
    
    For our purposes, we compute it as: for each face of the convex hull of the points,
    find the circumcenter of that face (on the sphere), and check the distance to the
    nearest point. The maximum over all such circumcenters (and the points themselves)
    gives the covering radius.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Compute all pairwise angular distances
    # cos_theta = dot product
    cos_theta = np.clip(pts @ pts.T, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # The covering radius is the maximum, over all points on the sphere, of the distance
    # to the nearest site. This is achieved at Voronoi vertices.
    # A simpler approach: sample the sphere densely? No, too slow.
    # Better: use the fact that the covering radius = max circumradius of spherical triangles
    # formed by 3 points where the triangle is empty (no other point inside).
    
    # Use Delaunay triangulation on the sphere via 3D convex hull
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
    except Exception:
        # Fallback: use a grid-based estimate
        # Create a dense grid on the sphere
        n_grid = 200
        theta_grid = np.linspace(0, np.pi, n_grid)
        phi_grid = np.linspace(0, 2*np.pi, n_grid)
        TH, PH = np.meshgrid(theta_grid, phi_grid)
        X = np.sin(TH) * np.cos(PH)
        Y = np.sin(TH) * np.sin(PH)
        Z = np.cos(TH)
        grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # For each grid point, find nearest site
        cos_dist = grid_pts @ pts.T
        nearest_dist = np.arccos(np.clip(cos_dist.max(axis=1), -1.0, 1.0))
        return np.max(nearest_dist) * 180.0 / np.pi
    
    # For each triangular face of the convex hull, compute the circumcenter
    # The circumcenter of a spherical triangle is the point on the sphere equidistant
    # from the three vertices.
    max_radius = 0.0
    for simplex in simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Find circumcenter: point x on unit sphere such that |x-a|=|x-b|=|x-c|
        # This means x·a = x·b = x·c
        # So x·(a-b) = 0 and x·(a-c) = 0
        # x is orthogonal to (a-b) and (a-c), so x is parallel to (a-b) x (a-c)
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-12:
            continue
        x = cross / cross_norm
        
        # Check if x is on the correct side (outside the hull, i.e., in the empty cap)
        # The circumcenter should be in the direction away from the interior
        # Check: the distance from x to a, b, c should be the same
        d_a = np.arccos(np.clip(np.dot(x, a), -1.0, 1.0))
        d_b = np.arccos(np.clip(np.dot(x, b), -1.0, 1.0))
        d_c = np.arccos(np.clip(np.dot(x, c), -1.0, 1.0))
        
        # Check if this circumcenter is valid: all other points should be at distance >= d
        # from x (i.e., the cap centered at x with radius d is empty)
        cos_x = np.clip(pts @ x, -1.0, 1.0)
        dists = np.arccos(cos_x)
        min_dist_to_other = np.min(dists)
        
        # The circumcenter is valid if the nearest point to x is one of a, b, c
        # and the distance is consistent
        if min_dist_to_other < d_a - 1e-9 or min_dist_to_other < d_b - 1e-9 or min_dist_to_other < d_c - 1e-9:
            # The circumcenter is inside the cell, so it's a valid Voronoi vertex
            radius = np.max([d_a, d_b, d_c])
            max_radius = max(max_radius, radius)
    
    # Also check the points themselves
    # The covering radius is at least the maximum distance from any point to its nearest neighbor
    # But actually, we need to check all Voronoi vertices
    # The above loop should cover all relevant circumcenters
    
    # Convert to degrees
    return max_radius * 180.0 / np.pi

def _smooth_covering_radius(pts, beta=100.0):
    """
    Smooth approximation of the covering radius using soft-max.
    For each point y on a dense grid, compute the soft-max of distances to nearest sites.
    """
    pts = _normalize(pts)
    n = len(pts)
    
    # Create a moderately dense grid
    n_grid = 150
    theta_grid = np.linspace(0, np.pi, n_grid)
    phi_grid = np.linspace(0, 2*np.pi, n_grid)
    TH, PH = np.meshgrid(theta_grid, phi_grid)
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Compute distances from grid points to all sites
    cos_dist = np.clip(grid_pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(cos_dist)  # shape (n_grid^2, n)
    
    # Soft-max approximation of the max
    # f(y) = (1/beta) * log(sum(exp(-beta * d_i(y))))  -> this is soft-min
    # We want soft-max of d_i(y)
    # g(y) = (1/beta) * log(sum(exp(beta * d_i(y))))
    # But this is the max over sites, not min. We want min over sites.
    # The covering radius is max_y min_i d(y, p_i)
    # So for each y, compute min_i d(y, p_i), then max over y.
    
    min_dists = np.min(dists, axis=1)  # shape (n_grid^2,)
    # Smooth max over y
    max_radius = (1.0/beta) * np.log(np.sum(np.exp(beta * min_dists)))
    
    return max_radius * 180.0 / np.pi

def _repulsion_refine(pts, n_steps=1000, dt=0.001, cooling=0.999):
    """
    Repulsion-based refinement: each point is pushed away from nearby points.
    """
    pts = _normalize(pts).copy()
    n = len(pts)
    
    for step in range(n_steps):
        # Compute forces: repulsion from all other points
        forces = np.zeros_like(pts)
        for i in range(n):
            # Distance to all other points
            cos_d = np.clip(pts @ pts[i], -1.0, 1.0)
            dists = np.arccos(cos_d)
            # Force magnitude: strong repulsion for close points
            # Use a potential like 1/d^2 or exp(-d)
            force_mag = np.exp(-dists / 0.1)
            force_mag[i] = 0  # No self-force
            # Direction: away from other points
            # The gradient of the distance on the sphere
            for j in range(n):
                if i == j:
                    continue
                # Move away from point j
                # The direction of steepest ascent of distance from j at point i
                # is (p_i - (p_i·p_j) p_j) normalized
                proj = np.dot(pts[i], pts[j])
                grad = pts[i] - proj * pts[j]
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-12:
                    forces[i] += force_mag[j] * grad / grad_norm
        
        # Update positions
        pts += dt * forces
        pts = _normalize(pts)
        dt *= cooling
    
    return pts

def _lloyd_iterate(pts, n_iters=100):
    """
    Lloyd's algorithm for spherical k-means: 
    1. Compute Voronoi cells (spherical)
    2. Move each point to the centroid of its cell
    """
    pts = _normalize(pts).copy()
    n = len(pts)
    
    for _ in range(n_iters):
        # Assign each point on a grid to its nearest site
        # Then compute the centroid of each cell
        n_grid = 200
        theta_grid = np.linspace(0, np.pi, n_grid)
        phi_grid = np.linspace(0, 2*np.pi, n_grid)
        TH, PH = np.meshgrid(theta_grid, phi_grid)
        X = np.sin(TH) * np.cos(PH)
        Y = np.sin(TH) * np.sin(PH)
        Z = np.cos(TH)
        grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Assign to nearest site
        cos_dist = np.clip(grid_pts @ pts.T, -1.0, 1.0)
        labels = np.argmax(cos_dist, axis=1)
        
        # Compute centroids
        new_pts = np.zeros_like(pts)
        for i in range(n):
            mask = labels == i
            if np.sum(mask) > 0:
                centroid = np.mean(grid_pts[mask], axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-12:
                    new_pts[i] = centroid / centroid_norm
                else:
                    new_pts[i] = pts[i]
            else:
                new_pts[i] = pts[i]
        
        new_pts = _normalize(new_pts)
        # Check convergence
        if np.max(np.linalg.norm(new_pts - pts, axis=1)) < 1e-8:
            pts = new_pts
            break
        pts = new_pts
    
    return pts

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    t_start = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Phase 1: Repulsion-based refinement
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 60:
        pts = _repulsion_refine(pts, n_steps=500, dt=0.002, cooling=0.998)
        r = _covering_radius(pts)
        if r < best_radius:
            best_radius = r
            best_pts = pts.copy()
    
    # Phase 2: Lloyd's iteration
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 40:
        pts = _lloyd_iterate(pts, n_iters=50)
        r = _covering_radius(pts)
        if r < best_radius:
            best_radius = r
            best_pts = pts.copy()
    
    # Phase 3: Local optimization using L-BFGS-B
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - 20:
        # Use a parameterization: each point is (theta, phi)
        # Or use the 3D coordinates with normalization constraint
        
        # Flatten points
        x0 = best_pts.ravel()
        
        def objective(x):
            p = x.reshape(-1, 3)
            p = _normalize(p)
            return _smooth_covering_radius(p, beta=50.0)
        
        def gradient(x):
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = objective(x_plus)
                x_minus = x.copy()
                x_minus[i] -= eps
                f_minus = objective(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * eps)
            return grad
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 100, 'gtol': 1e-6})
            if result.success or result.fun < best_radius:
                p_opt = _normalize(result.x.reshape(-1, 3))
                r = _covering_radius(p_opt)
                if r < best_radius:
                    best_radius = r
                    best_pts = p_opt.copy()
        except Exception:
            pass
    
    # Phase 4: High-precision polish
    elapsed = time.time() - t_start
    if elapsed < TIME_LIMIT - SAFETY_MARGIN:
        # Try a few more Lloyd iterations with finer grid
        pts = best_pts.copy()
        for _ in range(20):
            n_grid = 300
            theta_grid = np.linspace(0, np.pi, n_grid)
            phi_grid = np.linspace(0, 2*np.pi, n_grid)
            TH, PH = np.meshgrid(theta_grid, phi_grid)
            X = np.sin(TH) * np.cos(PH)
            Y = np.sin(TH) * np.sin(PH)
            Z = np.cos(TH)
            grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            
            cos_dist = np.clip(grid_pts @ pts.T, -1.0, 1.0)
            labels = np.argmax(cos_dist, axis=1)
            
            new_pts = np.zeros_like(pts)
            for i in range(N):
                mask = labels == i
                if np.sum(mask) > 0:
                    centroid = np.mean(grid_pts[mask], axis=0)
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm > 1e-12:
                        new_pts[i] = centroid / centroid_norm
                    else:
                        new_pts[i] = pts[i]
                else:
                    new_pts[i] = pts[i]
            
            new_pts = _normalize(new_pts)
            if np.max(np.linalg.norm(new_pts - pts, axis=1)) < 1e-10:
                pts = new_pts
                break
            pts = new_pts
        
        r = _covering_radius(pts)
        if r < best_radius:
            best_radius = r
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
