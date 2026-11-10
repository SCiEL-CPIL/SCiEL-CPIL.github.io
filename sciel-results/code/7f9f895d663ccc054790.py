"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _spherical_circumcenters(points, facets):
    """
    Compute circumcenters of spherical triangles defined by points and facets.
    Returns an array of shape (n_facets, 3) with normalized circumcenter directions.
    """
    a = points[facets[:, 0]]
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # For unit vectors a,b,c, the circumcenter direction is:
    # n = (b-a) x (c-a) = bxc + axb + cxa
    # which equals a x b + b x c + c x a
    ab = np.cross(a, b)
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    
    n = ab + bc + ca
    norms_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    centers = np.zeros_like(a)
    valid = norms_n.squeeze() > 1e-12
    
    if np.any(valid):
        centers[valid] = n[valid] / norms_n[valid]
        
    return centers

def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees using convex hull facets.
    The covering radius is the maximum circumradius of the spherical triangular facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        if len(facets) == 0:
            return 0.0, -1, None
            
        centers = _spherical_circumcenters(points, facets)
        
        # For each facet, the circumradius is the angular distance from center to any vertex
        a = points[facets[:, 0]]
        
        cos_R = np.sum(a * centers, axis=1)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        R = np.arccos(cos_R)
        max_idx = np.argmax(R)
        return np.degrees(R[max_idx]), max_idx, centers[max_idx]
    except Exception:
        return np.inf, -1, None

def _lloyd_step(points):
    """
    Perform one iteration of Spherical Lloyd's Algorithm.
    Moves each point to the centroid of its Voronoi cell (approximated by avg of adjacent facet circumcenters).
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        centers = _spherical_circumcenters(points, facets)
        
        new_points = np.zeros_like(points)
        counts = np.zeros(len(points))
        
        # Accumulate circumcenters for each vertex
        np.add.at(new_points, facets[:, 0], centers)
        np.add.at(new_points, facets[:, 1], centers)
        np.add.at(new_points, facets[:, 2], centers)
        
        np.add.at(counts, facets[:, 0], 1)
        np.add.at(counts, facets[:, 1], 1)
        np.add.at(counts, facets[:, 2], 1)
        
        # Normalize
        valid = counts > 0
        new_points[valid] /= counts[valid, None]
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        valid_norms = norms.squeeze() > 1e-12
        new_points[valid_norms] /= norms[valid_norms]
        
        # Keep original points for invalid cases
        new_points[~valid] = points[~valid]
        new_points[~valid_norms] = points[~valid_norms]
        
        return new_points
    except Exception:
        return points

def _energy_function(flat_points):
    """
    Riesz s=1 energy function for Coulomb repulsion.
    Minimizing this spreads points uniformly.
    """
    pts = flat_points.reshape(-1, 3)
    # Normalize to ensure they stay on sphere during optimization steps
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Calculate pairwise distances
    # dist^2 = |a|^2 + |b|^2 - 2 a.b = 2 - 2 a.b
    dot = np.dot(pts, pts.T)
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    
    # Energy is sum of 1/sqrt(dist_sq) = sum of 1/dist
    # We use s=1 (Coulomb)
    energy = np.sum(1.0 / np.sqrt(dist_sq))
    
    return energy

def _energy_gradient(flat_points):
    """
    Gradient of the Riesz s=1 energy function.
    """
    pts = flat_points.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    dot = np.dot(pts, pts.T)
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Gradient of 1/|x_i - x_j| w.r.t x_i is (x_i - x_j) / |x_i - x_j|^3
    # Sum over j for each i
    # term_ij = (x_i - x_j) / dist_ij^3
    # Since dist_ij^2 = 2 - 2 dot_ij, dist_ij^3 = (2 - 2 dot_ij)^1.5
    
    # We need to sum over j for each i
    # Let's compute the matrix of gradients
    # G[i, j] = (pts[i] - pts[j]) / dist[i, j]^3
    
    # Vectorized calculation
    # dist^3
    dist_cubed = dist ** 3
    np.fill_diagonal(dist_cubed, 1.0)
    
    # (pts[i] - pts[j])
    # We can use broadcasting: pts[:, None, :] - pts[None, :, :]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Divide by dist^3
    grad_terms = diff / dist_cubed[:, :, np.newaxis]
    
    # Sum over j (axis 1) to get gradient for each i
    grad = np.sum(grad_terms, axis=1)
    
    # Project gradient onto the tangent space of the sphere
    # grad_tangent = grad - (grad . pts) * pts
    proj = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - proj * pts
    
    return grad_tangent.flatten()

def _fibonacci_start(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_start(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    return pts

def _local_optimize_worst_facet(points, max_iters=100, step_size=0.05, rng=None, time_limit=None, start_time=None):
    """
    Targeted optimization: repeatedly identify the worst facet and perturb its vertices.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = points.copy()
    current_obj, worst_idx, worst_pt = _covering_radius_fast(current)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Recompute worst facet
        current_obj, worst_idx, worst_pt = _covering_radius_fast(current)
        if worst_idx < 0:
            break
            
        try:
            hull = ConvexHull(current)
            facets = hull.simplices
            worst_facet = facets[worst_idx]
        except Exception:
            break
            
        improved = False
        
        # Try perturbing the vertices of the worst facet
        for v_idx in worst_facet:
            for attempt in range(5):
                direction = rng.standard_normal(3)
                direction /= np.linalg.norm(direction)
                
                new_pts = current.copy()
                new_pts[v_idx] = current[v_idx] + step_size * direction
                new_pts[v_idx] /= np.linalg.norm(new_pts[v_idx])
                
                new_obj, _, _ = _covering_radius_fast(new_pts)
                
                if new_obj < current_obj:
                    current_obj = new_obj
                    current = new_pts
                    improved = True
                    break
                
        if not improved:
            # Try a smaller step
            step_size *= 0.9
            if step_size < 0.001:
                step_size = 0.05
                # Try moving a random point instead to escape local minima
                idx = rng.integers(0, len(current))
                direction = rng.standard_normal(3)
                direction /= np.linalg.norm(direction)
                new_pts = current.copy()
                new_pts[idx] = current[idx] + 0.02 * direction
                new_pts[idx] /= np.linalg.norm(new_pts[idx])
                new_obj, _, _ = _covering_radius_fast(new_pts)
                if new_obj < current_obj:
                    current_obj = new_obj
                    current = new_pts

    return current, current_obj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    initializations = [
        _fibonacci_start(N, offset=0.5),
        _fibonacci_start(N, offset=0.3),
        _fibonacci_start(N, offset=0.7),
        _random_start(N, rng),
        _random_start(N, rng),
    ]
    
    best_pts = initializations[0].copy()
    best_obj = _covering_radius_fast(best_pts)[0]
    
    # Phase 1: Energy Minimization (Coulomb/Riesz s=1)
    # This creates a very uniform distribution, which is a strong lower bound for covering radius.
    for init in initializations:
        if time.time() > deadline - 40.0:
            break
            
        # Optimize energy
        try:
            res = minimize(
                _energy_function, 
                init.flatten(), 
                jac=_energy_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 50}
            )
            opt_pts = res.x.reshape(-1, 3)
            # Normalize
            opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
            
            # Evaluate covering radius
            obj, _, _ = _covering_radius_fast(opt_pts)
            
            if obj < best_obj:
                best_obj = obj
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
            
    # Phase 2: Lloyd's Algorithm Refinement
    # Lloyd's algorithm directly minimizes the covering radius (or a proxy) by moving points to centroids.
    if time.time() < deadline - 20.0:
        current = best_pts.copy()
        for _ in range(50): # Limit iterations to stay within time
            if time.time() > deadline - 15.0:
                break
            current = _lloyd_step(current)
            obj, _, _ = _covering_radius_fast(current)
            if obj < best_obj:
                best_obj = obj
                best_pts = current.copy()
            else:
                # If it gets worse, stop Lloyd's and go to local opt
                break
                
    # Phase 3: Targeted Local Optimization
    if time.time() < deadline - 5.0:
        refined_pts, refined_obj = _local_optimize_worst_facet(
            best_pts, 
            max_iters=200, 
            step_size=0.05, 
            rng=rng,
            time_limit=deadline - time.time() - 2.0, 
            start_time=start_time
        )
        if refined_obj < best_obj:
            best_obj = refined_obj
            best_pts = refined_pts

    # Ensure output is valid
    pts_out = best_pts.copy()
    norms = np.linalg.norm(pts_out, axis=1)
    pts_out = pts_out / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_out[i] - pts_out[j]) < 1e-8:
                pts_out[j] += np.random.randn(3) * 0.01
                pts_out[j] /= np.linalg.norm(pts_out[j])
    
    return pts_out

# EVOLVE-BLOCK-END
