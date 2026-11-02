"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _initial_fibonacci():
    """Generate initial Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius(points):
    """
    Compute the covering radius: the maximum angular distance from any point
    on the sphere to the nearest configuration point.
    
    This equals the maximum circumradius of the facets of the convex hull,
    where each facet's circumcenter must be outside the cap (i.e., the facet
    is a valid facet with an empty cap).
    """
    points = np.asarray(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use nearest-neighbor based estimate
        # Compute pairwise distances
        diffs = points[:, None, :] - points[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        # Covering radius approx: max over vertices of distance to nearest neighbor / 2
        # But this is a lower bound. Use a grid-based estimate.
        # For robustness, return a large value
        return np.pi / 2
    
    # For each facet, compute the circumradius
    # The covering radius is the maximum over all facets of the angular distance
    # from the facet's circumcenter (projected to sphere) to the nearest point.
    
    # Get unique facets
    simplices = hull.simplices
    # Remove duplicate facets (each edge appears in two triangles for 3D hull)
    # Actually, ConvexHull.simplices gives each facet once
    
    max_circumradius = 0.0
    
    for simplex in simplices:
        pts = points[simplex]  # shape (k, 3)
        
        # Compute circumcenter of the facet
        # For a set of points on the unit sphere, the circumcenter is the point
        # equidistant from all facet vertices.
        # Solve: ||x - p_i||^2 = r^2 for all i, with |x| = 1
        # This means x · p_i = constant for all i
        # So x is in the null space of (p_i - p_0) for i=1..k-1
        # and |x| = 1
        
        if len(pts) < 3:
            # Degenerate facet
            continue
            
        # Set up linear system: x · (p_i - p_0) = 0 for i = 1, ..., k-1
        A = pts[1:] - pts[0]  # shape (k-1, 3)
        
        # Find null space of A
        try:
            U, s, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            continue
        
        # Null space vectors are the last rows of Vt
        # For 3D, null space dimension is 3 - rank(A)
        rank = np.sum(s > 1e-10)
        if rank < 2:
            # Degenerate: all points collinear
            continue
        
        # The null space is spanned by the last (3-rank) rows of Vt
        null_space = Vt[rank:]  # shape (3-rank, 3)
        
        if len(null_space) == 0:
            continue
        elif len(null_space) == 1:
            x = null_space[0]
            x = x / np.linalg.norm(x)
            # Two possible directions
            candidates = [x, -x]
        else:
            # 2D null space: circle of solutions
            # Need to find which one is on the unit sphere and maximizes distance
            # Actually, all points in the null space intersected with unit sphere
            # form a circle. We need the point farthest from the facet points.
            # This is more complex; approximate by checking a few points
            e1, e2 = null_space[0], null_space[1]
            e1 = e1 / np.linalg.norm(e1)
            e2 = e2 / np.linalg.norm(e2)
            # Orthonormalize
            e2 = e2 - np.dot(e1, e2) * e1
            e2 = e2 / np.linalg.norm(e2)
            candidates = []
            for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
                x = np.cos(theta) * e1 + np.sin(theta) * e2
                x = x / np.linalg.norm(x)
                candidates.append(x)
        
        for x in candidates:
            # Check that x is on the correct side (outside the cap)
            # The circumcenter should be such that all facet vertices are within
            # the cap centered at x
            # Compute angular distance from x to each facet vertex
            cos_dists = np.clip(np.dot(pts, x), -1.0, 1.0)
            ang_dists = np.arccos(cos_dists)
            
            # The circumradius is the max angular distance
            cr = np.max(ang_dists)
            
            # Verify empty cap: no other point should be closer to x than the facet vertices
            # i.e., x should be the circumcenter of this facet
            all_cos = np.clip(np.dot(points, x), -1.0, 1.0)
            all_ang = np.arccos(all_cos)
            nearest_dist = np.min(all_ang)
            
            # The facet is valid if the nearest point is one of the facet vertices
            # and the circumradius equals the distance to those vertices
            if np.abs(nearest_dist - cr) < 1e-6:
                max_circumradius = max(max_circumradius, cr)
            else:
                # This facet might not be a valid Delaunay facet
                # Still consider it
                max_circumradius = max(max_circumradius, cr)
    
    return max_circumradius


def _objective_flat(x):
    """Objective function: covering radius, with penalty for invalid configs."""
    points = x.reshape(N, 3)
    
    # Normalize
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10):
        return 10.0
    points = points / norms[:, None]
    
    # Check if all points are in a closed hemisphere
    # This would be invalid
    # Simple check: if all points have positive dot product with some direction
    # Use the fact that points are in a hemisphere iff there exists a vector v
    # such that v · p_i >= 0 for all i
    # This is equivalent to the origin being in the convex hull
    # Quick check: compute the convex hull and see if it contains the origin
    try:
        hull = ConvexHull(points)
        # Check if origin is inside the convex hull
        # The convex hull contains the origin iff the origin is a convex combination
        # of the hull vertices
        # Quick test: check if all facets have the origin on the same side
        # For each facet, the normal points outward. If origin is inside,
        # then for each facet, origin · normal <= 0
        normals = hull.equations[:, :-1]
        # hull.equations are of the form n·x + d <= 0 for points inside
        # The origin satisfies n·0 + d = d <= 0 iff d <= 0
        if np.all(hull.equations[:, -1] <= 1e-10):
            # Origin is inside, so not in a hemisphere
            pass
        else:
            # In a hemisphere - invalid
            return 10.0
    except Exception:
        return 10.0
    
    # Check for coincident points
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        return 10.0
    
    # Compute covering radius
    cr = _covering_radius(points)
    
    if not np.isfinite(cr):
        return 10.0
    
    return cr


def _local_optimize(points, max_time=60.0):
    """Run local optimization to improve the configuration."""
    start_time = time.time()
    
    x0 = points.flatten()
    best_x = x0.copy()
    best_obj = _objective_flat(x0)
    
    # Use Nelder-Mead with adaptive parameters
    options = {
        'maxiter': 5000,
        'xatol': 1e-10,
        'fatol': 1e-12,
        'adaptive': True,
    }
    
    while time.time() - start_time < max_time:
        result = minimize(
            _objective_flat,
            best_x,
            method='Nelder-Mead',
            options=options,
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
        
        # Check time
        if time.time() - start_time >= max_time:
            break
        
        # If no improvement, stop
        if result.fun >= best_obj * (1 - 1e-10):
            break
        
        # Small perturbation to escape local minima
        perturbation = 0.01 * np.random.randn(len(best_x))
        best_x = best_x + perturbation
    
    # Final normalization
    points = best_x.reshape(N, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points


def _polish(points, max_time=30.0):
    """High-precision polish pass with tighter tolerances."""
    start_time = time.time()
    
    x0 = points.flatten()
    best_x = x0.copy()
    best_obj = _objective_flat(x0)
    
    options = {
        'maxiter': 10000,
        'xatol': 1e-12,
        'fatol': 1e-14,
        'adaptive': True,
    }
    
    while time.time() - start_time < max_time:
        result = minimize(
            _objective_flat,
            best_x,
            method='Nelder-Mead',
            options=options,
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
        
        if time.time() - start_time >= max_time:
            break
        
        if result.fun >= best_obj * (1 - 1e-12):
            break
    
    points = best_x.reshape(N, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    Returns an array of shape (130, 3) of unit vectors.
    """
    total_deadline = 150.0
    start_time = time.time()
    
    # Generate initial configuration
    points = _initial_fibonacci()
    
    # Local optimization with time budget
    time_elapsed = time.time() - start_time
    remaining = total_deadline - time_elapsed - 30.0  # Reserve 30s for polish + safety
    
    if remaining > 10.0:
        points = _local_optimize(points, max_time=min(remaining, 90.0))
    
    # Polish pass
    time_elapsed = time.time() - start_time
    remaining = total_deadline - time_elapsed - 5.0  # Reserve 5s safety margin
    
    if remaining > 5.0:
        points = _polish(points, max_time=min(remaining, 30.0))
    
    # Final normalization and validation
    points = np.asarray(points)
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10):
        points = _initial_fibonacci()
    else:
        points = points / norms[:, None]
    
    # Ensure finite values
    if not np.all(np.isfinite(points)):
        points = _initial_fibonacci()
    
    return points
# EVOLVE-BLOCK-END
