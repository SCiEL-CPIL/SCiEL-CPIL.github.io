"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _spherical_centroid(pts):
    """Compute the spherical centroid (normalized sum) of a set of points."""
    # For small facets, normalized sum is a good approximation.
    # For larger facets, one might need iterative averaging, but normalized sum is standard for small steps.
    sum_pts = np.sum(pts, axis=0)
    norm = np.linalg.norm(sum_pts)
    if norm < 1e-10:
        return pts[0] # Fallback
    return sum_pts / norm


def _compute_circumradii(pts):
    """
    Compute the circumradius (in radians) for each facet of the convex hull.
    Returns:
        hull: ConvexHull object
        radii: array of circumradii corresponding to hull.simplices
    """
    pts = np.asarray(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None
        
    radii = np.zeros(len(hull.simplices))
    
    for i, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Circumcenter of spherical triangle
        # u is perpendicular to (a-b) and (a-c)
        u = np.cross(a - b, a - c)
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:
            radii[i] = 0.0
            continue
            
        u = u / u_norm
        
        # Ensure u is on the same side as the facet normal (outward)
        normal = np.cross(b - a, c - a)
        if np.dot(u, normal) < 0:
            u = -u
        
        # Compute circumradius
        dot_val = np.clip(np.dot(u, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        radii[i] = r
        
    return hull, radii


def _repulsive_force(pts, power=2.0):
    """
    Compute repulsive forces between all points.
    Force on i from j: k * (x_j - x_i) / d^power
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    dots = np.dot(pts, pts.T)
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    dists = np.arccos(dots)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Weights
    weights = np.zeros_like(dists)
    # Avoid division by zero
    safe_dists = np.where(dists < 1e-8, 1e-8, dists)
    weights[mask] = 1.0 / (safe_dists[mask] ** power)
    
    # Force matrix: F[i,j] = weights[i,j] * (pts[j] - pts[i])
    # This is not directly matrix multipliable in a simple way for vectorized sum without broadcasting issues.
    # F_i = sum_j w_ij (x_j - x_i) = (W @ X)_i - (sum_j w_ij) x_i
    # Let W be the weight matrix.
    # Sum_j w_ij = W @ ones
    
    W = weights
    sum_w = np.sum(W, axis=1, keepdims=True)
    
    # (W @ X) gives sum_j w_ij x_j
    weighted_sum = W @ pts
    
    forces = weighted_sum - sum_w * pts
    
    return forces


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Strategy:
    1. Initialize with Fibonacci spiral.
    2. Iteratively:
       a. Compute hull and circumradii.
       b. Identify the facet with the maximum circumradius (the "worst" hole).
       c. Apply a local "shrinking" force to the vertices of this facet, moving them towards the facet's centroid.
       d. Apply a global weak repulsive force to prevent points from collapsing and to maintain even distribution.
       e. Update points and normalize.
    """
    
    # Initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_cr = np.inf
    best_pts = pts.copy()
    
    # Parameters
    local_step = 0.05      # Step size for shrinking the worst facet
    global_step = 0.01     # Step size for global repulsion
    n_iter = 2000
    patience = 50
    no_improve = 0
    prev_cr = np.inf
    
    for it in range(n_iter):
        hull, radii = _compute_circumradii(pts)
        
        if hull is None:
            break
            
        current_cr = np.max(radii)
        
        if current_cr < best_cr:
            best_cr = current_cr
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > patience:
            break
            
        if current_cr < 0.32: # ~18.3 degrees
            break
            
        # 1. Global Repulsion
        # Weak repulsion to keep points spread out
        f_global = _repulsive_force(pts, power=2.0)
        
        # 2. Local Shrinking of Worst Facet
        # Find the index of the max radius
        idx_max = np.argmax(radii)
        worst_simplex = hull.simplices[idx_max]
        
        f_local = np.zeros_like(pts)
        
        # Move vertices towards centroid
        facet_pts = pts[worst_simplex]
        centroid = _spherical_centroid(facet_pts)
        
        for k, idx in enumerate(worst_simplex):
            # Vector from point to centroid
            # We want to move towards centroid.
            # Step direction: centroid - pts[idx]
            # Project onto tangent plane? 
            # Simple update: pts[idx] += step * (centroid - pts[idx])
            # Then normalize.
            
            # To make it a force field, we can just add this vector to the force.
            f_local[idx] += local_step * (centroid - pts[idx])
            
        # Combine forces
        # The local force is much stronger than global to ensure the worst hole closes
        total_force = f_global * global_step + f_local
        
        # Update
        pts_new = pts + total_force
        
        # Normalize
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        # Handle zero norms
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts_new = pts_new / norms
        
        pts = pts_new
        
    return best_pts
# EVOLVE-BLOCK-END
