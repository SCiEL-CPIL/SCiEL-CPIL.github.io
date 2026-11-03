"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    Uses the convex hull facets: for each facet, compute the circumradius (angular distance
    from facet centroid to farthest vertex), and the covering radius is the max over all facets.
    Actually, the covering radius is the maximum over the sphere of the distance to the nearest point.
    For a set of points on the sphere, the covering radius equals the maximum circumradius of the
    facets of the convex hull (Voronoi cells on the sphere).
    
    A simpler and more robust approach: use a fine grid on the sphere and compute the max angular
    distance to the nearest point. But this can be slow.
    
    Better: use the fact that the covering radius is the smallest r such that caps of radius r
    around the points cover the sphere. We can compute this by finding the maximum, over a dense
    set of test points, of the angular distance to the nearest configuration point.
    
    For efficiency and accuracy, we'll use a combination: compute the Voronoi vertices (circumcenters
    of facets) using scipy's ConvexHull, then for each facet, compute the angular circumradius.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial import cKDTree
    
    # Normalize to be safe
    pts = _normalize(pts)
    
    # Try to compute convex hull
    try:
        hull = ConvexHull(pts)
        # For each facet (triangle), compute the circumradius
        # The circumradius of a spherical triangle is related to its planar circumradius
        # but we need the angular circumradius.
        
        # Alternative: use a dense sampling approach which is more robust
        # Generate a fine icosahedron-based grid or Fibonacci spiral for testing
        n_test = 20000
        i_test = np.arange(n_test) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_test
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(1.0 - z_test * z_test)
        test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        
        # Compute angular distances from each test point to the nearest configuration point
        # Angular distance: arccos(clamp(dot, -1, 1))
        # For efficiency, compute dot products in batches
        max_dist = 0.0
        batch_size = 5000
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            batch = test_pts[start:end]
            # dot product: (batch_size, N)
            dots = batch @ pts.T
            # clamp to [-1, 1]
            dots = np.clip(dots, -1.0, 1.0)
            # angular distance
            dists = np.arccos(dots)
            # min over N points
            min_dists = np.min(dists, axis=1)
            # max over batch
            batch_max = np.max(min_dists)
            if batch_max > max_dist:
                max_dist = batch_max
        return np.degrees(max_dist)
    except Exception:
        # Fallback: coarse grid
        n_test = 5000
        i_test = np.arange(n_test) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_test
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(1.0 - z_test * z_test)
        test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.degrees(np.max(min_dists))


def _objective_and_grad(flat_pts):
    """Objective: weighted sum of squared deviations of facet circumradii from their mean.
    This enforces equiangularity. We use a smooth approximation via the dense grid method.
    """
    pts = _normalize(flat_pts.reshape(N, 3))
    
    # Use a moderate number of test points for speed during optimization
    n_test = 5000
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # Compute angular distances
    dots = test_pts @ pts.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_test, N)
    
    # For each test point, find nearest point and its distance
    min_dists = np.min(dists, axis=1)
    
    # The covering radius is max(min_dists), but for a smooth objective,
    # we use the mean of squared deviations of min_dists from their mean.
    # This encourages all Voronoi cells to have similar sizes (equiangularity).
    mean_dist = np.mean(min_dists)
    # Also add a small penalty for the max to ensure we're minimizing covering radius
    # Use a soft-max approximation for the max
    beta = 50.0
    log_sum_exp = np.log(np.sum(np.exp(beta * min_dists))) / beta
    
    # Objective: variance of distances + soft max
    var = np.mean((min_dists - mean_dist) ** 2)
    obj = var + 0.1 * log_sum_exp
    
    # Gradient via finite differences would be too slow; use automatic differentiation
    # But since we can't easily do that, we'll use a simpler approach:
    # Just return the objective and let scipy handle gradients via finite differences
    # This will be slow, so let's use a smarter approach.
    
    return obj


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using gradient descent on the equiangularity objective
    # Use L-BFGS-B with finite difference gradients (slow but robust)
    
    def _obj(flat):
        pts = _normalize(flat.reshape(N, 3))
        n_test = 3000
        i_test = np.arange(n_test) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_test
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(1.0 - z_test * z_test)
        test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        
        mean_dist = np.mean(min_dists)
        var = np.mean((min_dists - mean_dist) ** 2)
        beta = 30.0
        log_sum_exp = np.log(np.sum(np.exp(beta * min_dists))) / beta
        return var + 0.2 * log_sum_exp
    
    flat0 = pts0.flatten()
    
    # Use Nelder-Mead for robustness (no gradient needed)
    res = minimize(_obj, flat0, method='Nelder-Mead', 
                   options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10})
    
    pts_opt = _normalize(res.x.reshape(N, 3))
    
    # Final refinement: local search to minimize the actual covering radius
    # Use a few iterations of moving points toward their Voronoi centroids
    for _ in range(20):
        # For each point, find its Voronoi cell (points on sphere closer to it than to any other)
        # and move it toward the centroid of its cell
        n_test = 10000
        i_test = np.arange(n_test) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_test
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(1.0 - z_test * z_test)
        test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        
        dots = test_pts @ pts_opt.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        nearest = np.argmin(dists, axis=1)
        
        # For each point, compute the centroid of its Voronoi cell
        for j in range(N):
            mask = (nearest == j)
            if np.sum(mask) > 0:
                centroid = np.mean(test_pts[mask], axis=0)
                norm_c = np.linalg.norm(centroid)
                if norm_c > 1e-10:
                    centroid = centroid / norm_c
                    # Move point toward centroid with small step
                    pts_opt[j] = pts_opt[j] + 0.1 * (centroid - pts_opt[j])
        
        pts_opt = _normalize(pts_opt)
    
    return pts_opt
# EVOLVE-BLOCK-END
