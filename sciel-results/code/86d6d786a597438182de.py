import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(pts):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the exact covering radius using ConvexHull.
    Returns radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # Spherical circumcenter
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            c_s = normal / norm
            
            # Check if circumcenter is inside the spherical triangle
            # If the triangle is acute, the circumcenter is the center of the largest empty cap.
            # If obtuse, the center is the midpoint of the longest edge.
            
            # Distance from circumcenter to vertices
            d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
            d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
            
            # Edge midpoints and their distances to nearest vertex (which is half the edge length)
            e12 = v1 + v2
            e12_norm = np.linalg.norm(e12)
            if e12_norm > 1e-10:
                m12 = e12 / e12_norm
                d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
            else:
                d_e12 = 0.0
                
            e23 = v2 + v3
            e23_norm = np.linalg.norm(e23)
            if e23_norm > 1e-10:
                m23 = e23 / e23_norm
                d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
            else:
                d_e23 = 0.0

            e31 = v3 + v1
            e31_norm = np.linalg.norm(e31)
            if e31_norm > 1e-10:
                m31 = e31 / e31_norm
                d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
            else:
                d_e31 = 0.0

            # Determine local radius
            # If the circumcenter is "inside" (acute triangle), the radius is the distance to any vertex.
            # For a spherical triangle, if all angles are < 90, circumcenter is inside.
            # A simpler check: if d1, d2, d3 are all <= pi/2, it's acute/right.
            # Actually, for the covering radius of the facet, we need the max distance from any point in the facet to the nearest vertex.
            # If acute: max(d1, d2, d3) (which are equal).
            # If obtuse: max(d_e12, d_e23, d_e31) (specifically the one corresponding to the longest edge).
            
            # A robust way: The covering radius of the facet is the minimum radius R such that the three caps of radius R centered at v1, v2, v3 cover the facet.
            # This is equivalent to the max of the distances from the "worst point" to the nearest vertex.
            
            # Heuristic check often used:
            # If the triangle is obtuse, the circumcenter is outside. The worst point is the midpoint of the longest side.
            # If acute, the worst point is the circumcenter.
            
            # Let's check if the circumcenter is on the same side of the edges as the interior.
            # Simpler: if any of d1, d2, d3 > pi/2, it's obtuse.
            if max(d1, d2, d3) <= np.pi / 2.0:
                local_rad = max(d1, d2, d3)
            else:
                local_rad = max(d_e12, d_e23, d_e31)
            
            if local_rad > max_rad:
                max_rad = local_rad
        return max_rad
    except Exception:
        return np.pi

def _energy(pts):
    """
    Smooth proxy energy for covering radius.
    We want points to be far apart. 
    Minimizing sum(1/d^2) encourages uniform distribution.
    We also add a penalty for points being too close to avoid singularities.
    """
    # Compute pairwise distances
    # d_ij = 2 * sin(theta_ij / 2)
    # theta_ij = arccos(dot)
    # To avoid arccos, we can use dot products directly.
    # Let's use the squared chord distance: |pi - pj|^2 = 2 - 2 dot(pi, pj).
    # Minimizing sum(1 / (2 - 2 dot)^alpha) is equivalent to maximizing distances.
    
    # Efficient calculation using matrix multiplication
    # dots = pts @ pts.T
    # dists_sq = 2 - dots
    # dists_sq[diagonal] = 1e-10
    
    # However, this is O(N^2). N=50 is small, so it's fine.
    
    dots = pts @ pts.T
    # Set diagonal to large value to ignore self-interaction
    np.fill_diagonal(dots, 1.0) 
    # dists_sq = 2.0 - dots
    # To avoid division by zero if points coincide, clip
    dists_sq = np.clip(2.0 - dots, 1e-6, None)
    
    # Sum of inverse squared distances
    # This strongly repels close points
    energy = np.sum(1.0 / (dists_sq ** 2))
    
    # Add a small term to encourage spreading out globally (maximize min distance)
    # The 1/d^2 term is sufficient for this purpose in many contexts.
    return energy

def _energy_grad(pts):
    """
    Gradient of the energy function.
    E = sum_{i!=j} (2 - 2 p_i . p_j)^-2
    dE/dp_i = sum_{j!=i} -2 * (2 - 2 p_i . p_j)^-3 * (-2 * p_j)
            = sum_{j!=i} 4 * p_j / (2 - 2 p_i . p_j)^3
    """
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    dists_sq = np.clip(2.0 - dots, 1e-6, None)
    
    # We want dE/dp_i
    # Term for pair (i, j) in E is (2 - 2 p_i . p_j)^-2
    # Derivative w.r.t p_i is -2 * (2 - 2 p_i . p_j)^-3 * (-2 p_j) = 4 p_j / (2 - 2 p_i . p_j)^3
    
    # Matrix form:
    # Let M be matrix where M[i,j] = 4 / (2 - 2 p_i . p_j)^3
    # Then grad_i = sum_j M[i,j] p_j = M @ pts
    # Note: M is symmetric.
    
    # Avoid division by zero on diagonal (already handled by fill_diagonal)
    M = 4.0 / (dists_sq ** 3)
    np.fill_diagonal(M, 0.0)
    
    grad = M @ pts
    
    # Project gradient to tangent space of sphere
    # grad_tangent = grad - (grad . p_i) p_i
    # (grad . p_i) is row-wise dot product
    grad_dot_p = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - grad_dot_p * pts
    
    return grad_tangent

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_radius = np.inf
    
    # Run multiple restarts
    # We have ~115s. Each minimize call might take a few seconds.
    # Let's try 5-10 restarts.
    
    np.random.seed(42)
    
    # Initial guess: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for restart in range(20):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Perturb the initial guess for restarts > 0
        if restart == 0:
            x0 = points_init.copy()
        else:
            x0 = points_init.copy()
            # Add random perturbation
            noise = np.random.normal(0, 0.5, x0.shape)
            x0 = x0 + noise
            x0 = _normalize(x0)
            
        # Run optimization
        # Use L-BFGS-B with gradient
        # We minimize the energy
        try:
            res = minimize(
                _energy, 
                x0, 
                method='L-BFGS-B', 
                jac=_energy_grad,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_pts = _normalize(res.x)
            
            # Evaluate true covering radius
            rad = _covering_radius(opt_pts)
            
            if rad < best_radius:
                best_radius = rad
                best_points = opt_pts.copy()
                
        except Exception as e:
            pass # Skip failed restarts

    # If no optimization succeeded, return initial
    if best_points is None:
        return points_init
        
    return best_points

# EVOLVE-BLOCK-END
