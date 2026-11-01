"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import warnings

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    Uses SphericalVoronoi to find the farthest point on each facet from its nearest site.
    For a spherical Voronoi cell (great-circle polygon), the farthest point from the
    site is either at a vertex of the cell or at the midpoint of an edge (if the cell
    is large enough). More robustly, the covering radius is the maximum over all Voronoi
    vertices of the distance to the nearest site, but actually the farthest point in each
    cell from the site is a vertex of the Voronoi diagram (the circumcenter of the
    corresponding Delaunay triangle).
    
    Actually, the covering radius is the maximum, over all points y on the sphere, of
    min_i ||y - x_i||. This is equal to the maximum circumradius of the Delaunay
    triangles (spherical). The circumcenter of each Delaunay triangle is a Voronoi vertex.
    So we can compute all Voronoi vertices and find the max distance to nearest site.
    """
    try:
        sv = SphericalVoronoi(pts)
        # Voronoi vertices are the circumcenters of Delaunay triangles
        # The covering radius is the max over Voronoi vertices of distance to nearest site
        # But actually, the farthest point in each Voronoi cell from the site is a vertex
        # of that cell. The set of all Voronoi vertices is sv.vertices.
        # For each Voronoi vertex, find the nearest site and compute angular distance.
        
        verts = sv.vertices  # shape (M, 3), normalized
        if len(verts) == 0:
            # Fallback: use a dense grid
            return _covering_radius_grid(pts)
        
        # For each vertex, find min distance to any site
        # dist^2 = 2 - 2 * (verts @ pts.T)
        dot = verts @ pts.T  # (M, N)
        min_cos = np.max(dot, axis=1)  # max cos = min angle
        min_cos = np.clip(min_cos, -1.0, 1.0)
        angles = np.arccos(min_cos)
        return np.max(angles)
    except Exception:
        return _covering_radius_grid(pts)

def _covering_radius_grid(pts, n_lat=200, n_lon=400):
    """Fallback: compute covering radius using a dense spherical grid."""
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    
    # Convert to Cartesian
    x = np.cos(LAT) * np.cos(LON)
    y = np.cos(LAT) * np.sin(LON)
    z = np.sin(LAT)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find min distance to any site
    dot = grid @ pts.T  # (n_grid, N)
    min_cos = np.max(dot, axis=1)
    min_cos = np.clip(min_cos, -1.0, 1.0)
    angles = np.arccos(min_cos)
    return np.max(angles)

def _objective_and_grad(pts_flat):
    """
    Objective: covering radius in radians.
    Gradient: approximated via finite differences on the constraint manifold.
    We project points back to the unit sphere after each step.
    """
    pts = pts_flat.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    pts = pts / norms
    
    # Check for coincident points
    # If any two points are too close, add penalty
    min_dist = np.inf
    for i in range(N):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        d[i] = np.inf
        min_dist = min(min_dist, np.min(d))
    
    if min_dist < 1e-6:
        return 10.0, np.zeros_like(pts_flat)  # Large penalty
    
    # Compute covering radius
    cr = _covering_radius(pts)
    
    # Gradient via finite differences (central differences, step size h)
    h = 1e-5
    grad = np.zeros_like(pts_flat)
    for i in range(len(pts_flat)):
        pts_plus = pts_flat.copy()
        pts_minus = pts_flat.copy()
        pts_plus[i] += h
        pts_minus[i] -= h
        
        # Reshape and normalize
        p_plus = pts_plus.reshape(N, 3)
        p_minus = pts_minus.reshape(N, 3)
        
        n_plus = np.linalg.norm(p_plus, axis=1, keepdims=True)
        n_minus = np.linalg.norm(p_minus, axis=1, keepdims=True)
        n_plus = np.where(n_plus < 1e-10, 1.0, n_plus)
        n_minus = np.where(n_minus < 1e-10, 1.0, n_minus)
        p_plus = p_plus / n_plus
        p_minus = p_minus / n_minus
        
        # Check for coincident points in perturbed configs
        min_d_plus = np.inf
        min_d_minus = np.inf
        for j in range(N):
            d_p = np.linalg.norm(p_plus[j] - p_plus, axis=1)
            d_p[j] = np.inf
            min_d_plus = min(min_d_plus, np.min(d_p))
            
            d_m = np.linalg.norm(p_minus[j] - p_minus, axis=1)
            d_m[j] = np.inf
            min_d_minus = min(min_d_minus, np.min(d_m))
        
        if min_d_plus < 1e-6 or min_d_minus < 1e-6:
            grad[i] = 0.0
            continue
        
        cr_plus = _covering_radius(p_plus)
        cr_minus = _covering_radius(p_minus)
        grad[i] = (cr_plus - cr_minus) / (2 * h)
    
    return cr, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return pts / norms

def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    Uses gradient descent with finite-difference gradients on the spherical constraint.
    """
    np.random.seed(42)
    
    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random perturbation to break symmetry
    pts = pts + np.random.normal(0, 0.01, (N, 3))
    pts = _project_to_sphere(pts)
    
    # Use L-BFGS-B with finite-difference gradients
    # We optimize in R^{N*3} with the constraint that points are on the unit sphere.
    # We handle the constraint by projecting after each step.
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Try a few random restarts
    for restart in range(3):
        if restart > 0:
            # New random start
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.uniform(0, 2*np.pi, N)
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts = pts + np.random.normal(0, 0.1, (N, 3))
            pts = _project_to_sphere(pts)
        
        # Gradient descent with adaptive step size
        lr = 0.01
        for step in range(500):
            cr, grad = _objective_and_grad(pts.ravel())
            
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
            
            # Project gradient to tangent space
            # The constraint is ||p_i||^2 = 1 for each i
            # Gradient on the manifold: grad - (grad . p_i) p_i for each point
            grad_pts = grad.reshape(N, 3)
            proj_grad = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
            
            # Update
            pts_new = pts - lr * proj_grad
            pts_new = _project_to_sphere(pts_new)
            
            # Check if we improved
            cr_new = _covering_radius(pts_new)
            if cr_new < cr:
                pts = pts_new
                # Reduce learning rate slightly
                lr *= 0.99
            else:
                # Halve learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
    
    # Final polish with a finer grid-based local search
    # Try small random perturbations
    for _ in range(200):
        idx = np.random.randint(N)
        direction = np.random.normal(0, 1, 3)
        direction = direction / np.linalg.norm(direction)
        # Tangent direction
        tang = direction - np.dot(direction, best_pts[idx]) * best_pts[idx]
        tang_norm = np.linalg.norm(tang)
        if tang_norm < 1e-10:
            continue
        tang = tang / tang_norm
        
        for delta in [0.001, -0.001, 0.005, -0.005]:
            new_pts = best_pts.copy()
            new_pts[idx] = best_pts[idx] + delta * tang
            new_pts = _project_to_sphere(new_pts)
            
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = new_pts
    
    return best_pts
# EVOLVE-BLOCK-END
