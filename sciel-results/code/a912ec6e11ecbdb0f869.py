"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Global deadline
_DEADLINE = None
_BEST_SCORE = None
_BEST_POINTS = None


def _elapsed():
    return time.time() - _DEADLINE[0] if _DEADLINE else 0.0


def _time_left():
    return _DEADLINE[1] - time.time() if _DEADLINE else 120.0


def _covering_radius(points):
    """Compute covering radius in degrees for points on unit sphere."""
    # Use convex hull to find facets, then compute circumradius of each facet
    # But for robustness, also sample the sphere densely
    # Actually, the covering radius is the maximum over the sphere of min distance to any point.
    # We can approximate by sampling, but for accuracy we need the true max.
    
    # The true covering radius equals the maximum circumradius of the spherical facets
    # of the convex hull of the points (when projected to the sphere).
    
    # Alternative: use the fact that the covering radius is the radius of the largest
    # empty cap. This can be found by checking all Voronoi vertices on the sphere.
    
    # For practical purposes, we'll use a dense sampling approach combined with
    # checking the circumcenters of hull facets.
    
    # First, compute the convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback to dense sampling
        return _covering_radius_sample(points, 500)
    
    # For each facet (triangle in 3D), find the point on the sphere farthest from all N points
    # that lies in the spherical triangle corresponding to that facet.
    # This is the circumcenter of the spherical triangle if it lies inside, 
    # otherwise it's on the boundary.
    
    # Simpler approach: for each facet, compute the normal direction (circumcenter direction)
    # and check if it's valid (not inside any other cap)
    
    # Actually, the covering radius is the max over all facets of the angular distance
    # from the facet's circumcenter (on sphere) to the nearest point, BUT only if the
    # circumcenter lies within the spherical facet. If not, the max is on an edge or vertex.
    
    # Let's use a hybrid: compute circumcenters of all hull facets, plus check edges.
    
    max_rad = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        # Compute circumcenter of the triangle (in 3D, the center of the sphere through 3 points)
        # For points on unit sphere, the circumcenter of the spherical triangle is the
        # normalized vector perpendicular to the plane of the triangle, pointing outward.
        
        # Plane normal:
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        
        # The circumcenter on the unit sphere is in the direction of n (outward)
        # Check if this point is actually the farthest point in this facet from all N points
        # by verifying it's not closer to some other point than to the facet's defining points.
        
        # Distance from circumcenter to each of the 3 points:
        d0 = np.arccos(np.clip(np.dot(n, tri[0]), -1, 1))
        d1 = np.arccos(np.clip(np.dot(n, tri[1]), -1, 1))
        d2 = np.arccos(np.clip(np.dot(n, tri[2]), -1, 1))
        facet_rad = max(d0, d1, d2)
        
        # Check if circumcenter is inside the spherical triangle:
        # It is inside if all three dot products n·tri[i] are >= cos(facet_rad) approximately,
        # but more precisely, the circumcenter is inside if the angles at each vertex are < pi.
        # A simpler check: the circumcenter is valid if it's farther from all other points
        # than from the 3 defining points.
        
        # Compute distance from circumcenter to all N points:
        dists = np.arccos(np.clip(points @ n, -1, 1))
        min_dist = np.min(dists)
        
        # If min_dist is achieved by one of the 3 points (approximately), then circumcenter is valid
        # Otherwise, the max is on an edge
        if min_dist + 1e-10 >= min(d0, d1, d2):
            # Circumcenter is the farthest point in this facet
            if min_dist > max_rad:
                max_rad = min_dist
        else:
            # Check edges of this triangle
            for i in range(3):
                for j in range(i+1, 3):
                    # Edge between tri[i] and tri[j]
                    # The farthest point on this edge from all N points
                    # is either the midpoint direction or an endpoint
                    mid = (tri[i] + tri[j]) / np.linalg.norm(tri[i] + tri[j]) if np.linalg.norm(tri[i]+tri[j]) > 1e-12 else tri[i]
                    dists_mid = np.arccos(np.clip(points @ mid, -1, 1))
                    if np.min(dists_mid) > max_rad:
                        max_rad = np.min(dists_mid)
    
    # Also do a dense sampling to catch any issues
    sampled_rad = _covering_radius_sample(points, 200)
    return max(max_rad, sampled_rad) * 180.0 / np.pi


def _covering_radius_sample(points, n_theta=200, n_phi=400):
    """Approximate covering radius by dense sampling."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to cartesian
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    samples = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each sample, find min distance to any point
    # Use batched computation
    # dot products: samples @ points.T gives (n_samples, N)
    dots = samples @ points.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)  # max dot = min angle
    min_angles = np.arccos(min_dots)
    return np.max(min_angles) * 180.0 / np.pi


def _optimize_step(points, max_iter=200, tol=1e-8):
    """Gradient-free local optimization using steepest descent on the sphere."""
    pts = points.copy()
    n = len(pts)
    
    for it in range(max_iter):
        if _time_left() < 5:
            break
        
        # Compute covering radius and identify worst region
        # For each point, compute the "Voronoi cell" contribution
        # Simple approach: for each point, compute the max distance to nearest other point
        # in its local neighborhood, then move it to reduce the worst case.
        
        # Compute pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1, 1)
        angles = np.arccos(dots)
        
        # For each point i, find the farthest point in its Voronoi cell
        # Approximate: the farthest point from i that is closer to i than to any other point
        # This is complex, so use a simpler heuristic:
        # Move each point in the direction that reduces the maximum local gap
        
        # Compute for each point the distance to its nearest neighbor
        np.fill_diagonal(angles, np.pi)
        min_dist = np.min(angles, axis=1)
        
        # Points with large min_dist are isolated; points with small min_dist are clustered
        # The covering radius is dominated by the largest gaps
        
        # Find the direction of the largest gap for each point:
        # For point i, the worst direction is the one that maximizes the minimum distance to all points
        # Approximate by moving away from the centroid of neighbors
        
        moved = False
        for i in range(n):
            if _time_left() < 2:
                break
            
            # Find neighbors (points within some distance)
            mask = angles[i] < np.pi * 0.3
            neighbors = np.where(mask)[0]
            if len(neighbors) < 2:
                continue
            
            # Centroid of neighbors
            centroid = pts[neighbors].mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-10:
                continue
            centroid = centroid / centroid_norm
            
            # Move point i away from centroid
            step_size = 0.01
            new_pt = pts[i] - step_size * centroid
            new_pt = new_pt / np.linalg.norm(new_pt)
            
            # Check if this improves the local covering
            old_local_rad = _local_covering_radius(pts, i)
            pts[i] = new_pt
            new_local_rad = _local_covering_radius(pts, i)
            
            if new_local_rad > old_local_rad + 1e-12:
                pts[i] = pts[i]  # revert
                # Try smaller step
                new_pt = pts[i] - step_size * 0.1 * centroid
                new_pt = new_pt / np.linalg.norm(new_pt)
                pts[i] = new_pt
                new_local_rad = _local_covering_radius(pts, i)
                if new_local_rad > old_local_rad + 1e-12:
                    pts[i] = pts[i]  # revert again
            else:
                moved = True
        
        if not moved:
            break
    
    return pts


def _local_covering_radius(points, center_idx, n_samples=50):
    """Estimate local covering radius around a point."""
    # Sample directions around points[center_idx]
    p = points[center_idx]
    
    # Generate random unit vectors near p
    # Use small perturbations
    rng = np.random.default_rng(42 + center_idx)
    perturbations = rng.normal(0, 0.1, size=(n_samples, 3))
    samples = p[None, :] + perturbations
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples / norms
    
    # Compute min distance from each sample to all points
    dots = samples @ points.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)


def _refine_with_grid(points, n_theta=150, n_phi=300):
    """Refine points using a grid-based approach to find and fix gaps."""
    pts = points.copy()
    
    # Create grid
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    samples = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find the nearest configuration point
    dots = samples @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    nearest_idx = np.argmax(dots, axis=1)
    
    # Find the worst grid point
    worst_idx = np.argmax(min_angles)
    worst_angle = min_angles[worst_idx]
    worst_sample = samples[worst_idx]
    
    # Move the nearest point toward the worst sample
    nearest_i = nearest_idx[worst_idx]
    direction = worst_sample - pts[nearest_i]
    dir_norm = np.linalg.norm(direction)
    if dir_norm > 1e-10:
        direction = direction / dir_norm
        step = min(0.05, worst_angle * 0.5)
        new_pt = pts[nearest_i] + step * direction
        new_pt = new_pt / np.linalg.norm(new_pt)
        pts[nearest_i] = new_pt
    
    return pts


def construct():
    global _DEADLINE, _BEST_SCORE, _BEST_POINTS
    
    _DEADLINE = [time.time(), time.time() + 145.0]  # 145s deadline with 5s margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Initial score
    score = _covering_radius(points)
    _BEST_SCORE = score
    _BEST_POINTS = points.copy()
    
    # Phase 1: Local optimization
    while _time_left() > 30:
        points = _optimize_step(points, max_iter=50)
        score = _covering_radius(points)
        if score < _BEST_SCORE - 1e-10:
            _BEST_SCORE = score
            _BEST_POINTS = points.copy()
        else:
            # Try grid-based refinement
            points = _refine_with_grid(points)
            score = _covering_radius(points)
            if score < _BEST_SCORE - 1e-10:
                _BEST_SCORE = score
                _BEST_POINTS = points.copy()
            else:
                # Try perturbation and restart
                if _time_left() < 10:
                    break
                rng = np.random.default_rng()
                perturbation = rng.normal(0, 0.05, size=points.shape)
                points = points + perturbation
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points = points / norms
                score = _covering_radius(points)
                if score < _BEST_SCORE - 1e-10:
                    _BEST_SCORE = score
                    _BEST_POINTS = points.copy()
    
    # Phase 2: Final high-precision polish
    if _time_left() > 10:
        points = _BEST_POINTS.copy()
        # Multiple refinement passes with finer grid
        for _ in range(3):
            if _time_left() < 5:
                break
            points = _refine_with_grid(points, n_theta=150, n_phi=300)
            score = _covering_radius(points)
            if score < _BEST_SCORE - 1e-10:
                _BEST_SCORE = score
                _BEST_POINTS = points.copy()
    
    # Ensure valid output
    points = _BEST_POINTS
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Final validation
    if not np.all(np.isfinite(points)):
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
    
    return points
# EVOLVE-BLOCK-END
