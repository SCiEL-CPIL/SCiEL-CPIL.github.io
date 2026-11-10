"""Spectral + local optimization for covering the sphere with 130 caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(points):
    """Compute covering radius in degrees using convex hull facets."""
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-12:
                return 90.0  # Invalid configuration
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere iff there exists a vector v such that v·x >= 0 for all x
    # This is equivalent to the origin being in the convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    # Check if origin is inside the convex hull
    # For each facet, check if origin is on the same side as the interior
    # A facet has outward normal n and equation n·x = d where d > 0 for points
    # Origin is inside iff n·0 - d <= 0 for all facets, i.e., d >= 0
    # Actually, for a convex hull, the facets' outward normals point away from interior
    # The equation is n·x = d where n is outward normal (unit) and d = n·p for any p on facet
    # Origin is inside iff n·0 <= d for all facets, i.e., d >= 0
    # But we need to be careful: if d < 0 for any facet, origin is outside (in that hemisphere)
    
    # Get facet normals and offsets
    # scipy ConvexHull: equations[i] = [n_x, n_y, n_z, d] where n·x + d = 0 for points on facet i
    # n is the outward normal (unit length), so n·x + d = 0 => n·x = -d
    # For a point p on the facet: n·p = -d
    # Origin: n·0 = 0, so 0 + d = d. If d < 0, then 0 < -d = n·p, meaning origin is on the interior side
    # Wait: the half-space containing the hull is n·x + d <= 0 (since n is outward)
    # Origin is inside iff 0 + d <= 0 iff d <= 0
    # If any d > 0, origin is outside that half-space, so origin is outside the hull
    
    equations = hull.equations  # shape (n_facets, 4)
    d_values = equations[:, 3]
    # Origin is inside iff all d_values <= 0
    if np.any(d_values > 1e-10):
        # Origin is outside the convex hull, so all points are in some closed hemisphere
        return 90.0
    
    # Compute covering radius from facets
    # For each facet, the farthest point from the facet (on the sphere) in the cap defined by that facet
    # The angular distance from a point y on the sphere to the facet plane is related to the distance
    # to the nearest point on the facet
    # For a facet with outward unit normal n and offset d (n·x + d = 0), the cap center is along n
    # The angular radius of the cap is arccos(-d) if d < 0 (since the facet is at distance -d from origin along -n)
    # Wait: the facet plane is n·x = -d. The distance from origin to this plane is |d|/||n|| = |d| (since n is unit)
    # Since n is outward and origin is inside, n·origin + d = d <= 0, so d <= 0
    # The plane is at distance -d from origin in the direction of -n... no, in the direction of n?
    # n·x = -d. If d < 0, then -d > 0, so the plane is at distance -d from origin in the direction of n
    # The cap centered at n (unit vector) has angular radius theta where cos(theta) = -d
    # Wait, let me reconsider. The facet is a triangle on the sphere. The cap that has this facet as its boundary
    # is centered at the direction of the outward normal n. The angular radius is the angle from n to any vertex of the facet.
    # For a vertex v (unit vector) on the facet: n·v = -d. So cos(angle) = n·v = -d.
    # So the angular radius is arccos(-d).
    
    # The covering radius is the maximum over all facets of arccos(-d)
    # But we need -d <= 1 for arccos to be valid
    covering_radius_rad = 0.0
    for d_val in d_values:
        cos_theta = -d_val
        if cos_theta > 1.0:
            cos_theta = 1.0
        if cos_theta < -1.0:
            cos_theta = -1.0
        theta = np.arccos(cos_theta)
        if theta > covering_radius_rad:
            covering_radius_rad = theta
    
    return np.degrees(covering_radius_rad)

def _energy(points):
    """Energy function: sum of exp(-k * min_pairwise_distance) + covering radius penalty."""
    # Compute all pairwise distances
    # Use dot products
    dots = points @ points.T
    np.fill_diagonal(dots, 2.0)
    # Clip to [-1, 1]
    dots = np.clip(dots, -1.0, 1.0)
    # Pairwise angular distances
    dists = np.arccos(dots)
    np.fill_diagonal(dists, 0.0)
    
    # For each point, find its nearest neighbor distance
    min_dists = np.min(dists, axis=1)
    
    # Energy: sum of exp(-alpha * min_dist) to encourage separation
    alpha = 5.0
    sep_energy = np.sum(np.exp(-alpha * min_dists))
    
    # Covering radius
    cr = _covering_radius(points)
    
    return sep_energy + 10.0 * cr

def _local_optimize(points, max_iters=2000, alpha_start=0.01, alpha_end=0.0001):
    """Local optimization using gradient descent on the sphere."""
    points = points.copy()
    N = len(points)
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        # Compute gradient of covering radius approximation
        # Use the energy function gradient
        # For simplicity, use a numerical gradient or an analytic approximation
        
        # Compute pairwise distances
        dots = points @ points.T
        np.fill_diagonal(dots, 2.0)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, 0.0)
        
        # Gradient of sum of exp(-alpha * min_dist_i) w.r.t. points
        # This is complex; use a simpler approach: repulsive forces
        # For each pair (i,j), force on i is proportional to (p_j - p_i) * exp(-alpha * dist_ij)
        # normalized to be tangent to the sphere
        
        alpha = alpha_start * (alpha_end / alpha_start) ** (it / max_iters)
        
        grad = np.zeros_like(points)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Force magnitude
                f_mag = np.exp(-alpha * d) / d
                # Direction: from j to i (repulsive)
                f_dir = points[i] - points[j]
                f_dir_norm = np.linalg.norm(f_dir)
                if f_dir_norm < 1e-15:
                    continue
                f_dir = f_dir / f_dir_norm
                grad[i] += f_mag * f_dir
        
        # Project gradient to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], points[i]) * points[i]
        
        # Update
        step = alpha * 0.1
        new_points = points + step * grad
        # Renormalize
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if improvement
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
        
        points = new_points
    
    return best_points, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Multiple restarts with local optimization
    np.random.seed(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            pts = points.copy()
        else:
            # Perturb the initial configuration
            pts = points + 0.1 * np.random.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Local optimization
        optimized_pts, cr = _local_optimize(pts, max_iters=1000)
        
        if cr < best_cr:
            best_cr = cr
            best_points = optimized_pts.copy()
        
        print(f"Restart {restart}: CR = {best_cr:.6f} degrees")
    
    # Final polish with smaller steps
    if time.time() < deadline - 10.0:
        polished_pts, cr = _local_optimize(best_points, max_iters=500, alpha_start=0.001, alpha_end=0.00001)
        if cr < best_cr:
            best_cr = cr
            best_points = polished_pts.copy()
    
    return best_points
# EVOLVE-BLOCK-END
