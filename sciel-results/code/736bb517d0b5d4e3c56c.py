"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START

def _compute_covering_radius(pts):
    """
    Compute the covering radius (in degrees) of a set of points on the unit sphere.
    The covering radius is the maximum, over all facets of the convex hull,
    of the circumradius of that facet (the angular distance from the facet's
    center to its vertices, projected onto the sphere).
    """
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, None]
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute max distance to nearest point on a grid
        # This is a rough approximation
        lat = np.linspace(-np.pi/2, np.pi/2, 200)
        lon = np.linspace(0, 2*np.pi, 400)
        LL, LLon = np.meshgrid(lat, lon, indexing='ij')
        grid_pts = np.column_stack([
            np.cos(LL) * np.cos(LLon),
            np.cos(LL) * np.sin(LLon),
            np.sin(LL)
        ]).reshape(-1, 3)
        
        # For each grid point, find nearest config point
        # Use chunked computation to save memory
        max_dist = 0.0
        chunk = 10000
        for i in range(0, len(grid_pts), chunk):
            gp = grid_pts[i:i+chunk]
            # dot product
            dots = gp @ pts.T  # (chunk, N)
            # clip for numerical safety
            dots = np.clip(dots, -1.0, 1.0)
            # angular distance = arccos(dot)
            dists = np.arccos(dots, axis=1)
            max_dist = max(max_dist, np.max(dists))
        
        return np.degrees(max_dist)
    
    max_circumradius = 0.0
    for simplex in hull.simplices:
        # simplex is a list of 3 vertex indices
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the circumcenter of the triangle on the sphere
        # The circumcenter is the point equidistant from all three vertices
        # On the unit sphere, this is the normalized cross product of (v0-v1) and (v0-v2)
        # Actually, for a triangle on the unit sphere, the circumcenter is:
        # normalize(v0 + v1 + v2) only if the triangle is small enough
        # More robustly: the circumcenter lies along the normal to the plane of the triangle
        # passing through the origin.
        
        # Normal to the triangle
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # The circumcenter is along the direction of n (or -n).
        # The circumradius (angular) is the angle from the circumcenter to any vertex.
        # We need to determine which sign of n gives the center "inside" the cap.
        # The center should be on the same side as the points relative to the plane.
        # For a convex hull facet, the outward normal points away from the hull.
        # The circumcenter of the facet (as a cap center) should be in the direction
        # that is "outside" the hull, i.e., along the outward normal.
        
        # Actually, for the covering radius, we want the point on the sphere that is
        # farthest from all points in the facet's triangle. That's the circumcenter
        # of the triangle, which is the point equidistant from all three vertices.
        
        # The circumcenter c satisfies: c·v0 = c·v1 = c·v2 = cos(theta)
        # where theta is the circumradius (angular distance).
        # c is along the normal n. Let c = alpha * n.
        # Then alpha * (n·v0) = cos(theta)
        # alpha = cos(theta) / (n·v0)
        # Also, |c| = 1, so alpha^2 = 1, meaning alpha = ±1.
        # Wait, that's not right. c is on the unit sphere, so |c|=1.
        # c = n or c = -n.
        # Let's check which one is the circumcenter.
        # The circumcenter is the point on the sphere equidistant from v0, v1, v2.
        # So we pick the sign of n that maximizes the dot product with the vertices
        # (if the triangle is small, the circumcenter is "above" the triangle).
        
        # Actually, for a spherical triangle, the circumcenter is the point on the sphere
        # that is equidistant from all three vertices. There are two such points (antipodal).
        # We want the one that is the center of the minimal cap covering the triangle.
        # That's the one with the smaller angular distance to the vertices.
        
        d0 = np.dot(n, v0)
        d1 = np.dot(n, v1)
        d2 = np.dot(n, v2)
        
        # If d0, d1, d2 are all positive, then n is the circumcenter direction.
        # Otherwise, -n is.
        if d0 + d1 + d2 > 0:
            c = n
        else:
            c = -n
        
        # Circumradius is the angular distance from c to any vertex
        cos_theta = np.dot(c, v0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        max_circumradius = max(max_circumradius, theta)
    
    return np.degrees(max_circumradius)


def _perturb(pts, sigma=0.05):
    """Add Gaussian perturbation and renormalize."""
    noise = np.random.randn(*pts.shape) * sigma
    new_pts = pts + noise
    norms = np.linalg.norm(new_pts, axis=1)
    new_pts = new_pts / norms[:, None]
    return new_pts


def _hill_climbing(pts, max_iters=5000, step_size=0.05, tol=1e-10):
    """
    Local optimization: move points to reduce the covering radius.
    Use a simple gradient-free approach: for each point, try small moves
    and keep the one that reduces the max circumradius of its neighboring facets.
    """
    pts = pts.copy()
    current_radius = _compute_covering_radius(pts)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Try moving point i in a random direction
            best_pts = pts.copy()
            best_radius = current_radius
            
            for _ in range(20):
                # Random perturbation of point i
                noise = np.random.randn(3) * step_size
                new_pts = pts.copy()
                new_pts[i] = pts[i] + noise
                new_pts[i] /= np.linalg.norm(new_pts[i])
                
                new_radius = _compute_covering_radius(new_pts)
                if new_radius < best_radius - 1e-12:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
            
            if best_radius < current_radius - 1e-12:
                pts = best_pts
                current_radius = best_radius
                improved = True
        
        if not improved:
            break
    
    return pts, current_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate initial configuration
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random initialization with some structure
            pts = np.random.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1)[:, None]
            # Add some perturbation to break symmetry
            pts = _perturb(pts, sigma=0.1)
        
        # Local optimization
        pts, radius = _hill_climbing(pts, max_iters=2000, step_size=0.03)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts

# EVOLVE-BLOCK-END
