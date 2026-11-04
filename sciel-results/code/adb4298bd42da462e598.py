"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and circumradius for each facet.
    Returns:
        hull: ConvexHull object or None
        radii: array of circumradii for each facet
        max_radius: maximum circumradius
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, np.pi
        
    radii = np.zeros(len(hull.simplices))
    max_radius = 0.0
    
    for idx, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii[idx] = 0.0
            continue
            
        # Normal vector to the facet
        n = cross / norm_cross
        
        # Ensure normal points outward (away from origin)
        if np.dot(n, a) < 0:
            n = -n
            
        d = np.dot(n, a)
        d = np.clip(d, -1.0, 1.0)
        r = np.arccos(d)
        radii[idx] = r
        
        if r > max_radius:
            max_radius = r
            
    return hull, radii, max_radius

def _smooth_objective_and_grad(x_flat, beta=50.0):
    """
    Smooth surrogate for the covering radius.
    Uses log-sum-exp to approximate the max of the circumradii.
    x_flat: flattened array of 3N coordinates.
    """
    pts = x_flat.reshape(N, 3)
    pts = _to_sphere(pts)
    
    hull, radii, max_r = _compute_hull_and_radii(pts)
    if hull is None:
        return 10.0, np.zeros_like(x_flat)
    
    if len(radii) == 0:
        return 10.0, np.zeros_like(x_flat)
        
    # Log-sum-exp approximation of max
    r_max = np.max(radii)
    if r_max == 0:
        return 0.0, np.zeros_like(x_flat)
        
    exp_terms = np.exp(beta * (radii - r_max))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return r_max, np.zeros_like(x_flat)
        
    obj_val = r_max + (1.0 / beta) * np.log(sum_exp)
    
    # Compute gradient
    grad_pts = np.zeros_like(pts)
    
    for idx, simplex in enumerate(hull.simplices):
        w = exp_terms[idx] / sum_exp
        if w < 1e-9:
            continue
            
        i, j, k = simplex
        a = pts[i]
        b = pts[j]
        c = pts[k]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            continue
            
        n = cross / norm_cross
        if np.dot(n, a) < 0:
            n = -n
            
        d = np.clip(np.dot(n, a), -1.0, 1.0)
        sin_r = np.sqrt(1.0 - d*d)
        if sin_r < 1e-6:
            sin_r = 1e-6
            
        g_vec = -n / sin_r
        
        grad_pts[i] += w * g_vec
        grad_pts[j] += w * g_vec
        grad_pts[k] += w * g_vec
        
    grad_x = np.zeros_like(x_flat)
    for i in range(N):
        p = pts[i]
        gp = grad_pts[i]
        grad_tangent = gp - np.dot(gp, p) * p
        grad_x[i*3:(i+1)*3] = grad_tangent
        
    return obj_val, grad_x

def _spherical_lloyd_iteration(pts, n_iters=10):
    """
    Spherical Lloyd relaxation: move each point to the centroid of its Voronoi cell
    (approximated by the centroid of its Delaunay neighbors on the sphere).
    """
    pts = _to_sphere(pts.copy())
    
    for _ in range(n_iters):
        if time.time() > _DEADLINE - 15:
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Build adjacency from hull simplices
        adj = [set() for _ in range(N)]
        for simplex in hull.simplices:
            i, j, k = simplex
            adj[i].add(j)
            adj[i].add(k)
            adj[j].add(i)
            adj[j].add(k)
            adj[k].add(i)
            adj[k].add(j)
            
        new_pts = np.zeros_like(pts)
        for i in range(N):
            neighbors = list(adj[i])
            if len(neighbors) == 0:
                new_pts[i] = pts[i]
            else:
                # Centroid of neighbors and self, projected to sphere
                centroid = pts[i] + sum(pts[j] for j in neighbors)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_pts[i] = centroid / norm
                else:
                    new_pts[i] = pts[i]
                    
        pts = _to_sphere(new_pts)
        
    return pts

def _repulsion_energy_and_grad(x_flat):
    """
    Compute repulsion energy and gradient.
    Energy is sum over pairs of (1 - dot)^-k.
    This encourages points to spread out.
    """
    pts = x_flat.reshape(N, 3)
    pts = _to_sphere(pts)
    
    # Calculate all pairwise dot products
    # D is N x N matrix of dot products
    D = np.dot(pts, pts.T)
    
    # We want to minimize sum_{i != j} (1 - D_ij)^-k
    # To avoid singularity at D=1 (coincident points), we clip D to 1 - eps
    eps = 1e-6
    D_clipped = np.clip(D, -1.0, 1.0 - eps)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    D_work = D_clipped[mask]
    
    # Energy: sum (1 - D)^-k
    k = 2.0
    terms = (1.0 - D_work) ** (-k)
    energy = np.sum(terms)
    
    # Gradient
    # d/dx_i sum_{j != i} (1 - x_i.x_j)^-k
    # = sum_{j != i} -k (1 - x_i.x_j)^(-k-1) * (-x_j)
    # = k * sum_{j != i} x_j / (1 - x_i.x_j)^(k+1)
    
    grad_pts = np.zeros_like(pts)
    
    # Vectorized gradient calculation
    # We need sum_j w_ij x_j for each i
    # w_ij = k / (1 - D_ij)^(k+1) for i != j
    
    D_for_grad = D_clipped.copy()
    np.fill_diagonal(D_for_grad, 0.0)
    
    weights = k * (1.0 - D_for_grad) ** (-(k+1))
    np.fill_diagonal(weights, 0.0)
    
    # grad_pts[i] = sum_j weights[i,j] * pts[j]
    grad_pts = np.dot(weights, pts)
    
    # Project to tangent space
    grad_x = np.zeros_like(x_flat)
    for i in range(N):
        p = pts[i]
        gp = grad_pts[i]
        grad_tangent = gp - np.dot(gp, p) * p
        grad_x[i*3:(i+1)*3] = grad_tangent
        
    return energy, grad_x

def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _to_sphere(pts)

def _random_points(seed):
    np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _to_sphere(pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    best_pts = None
    best_obj = np.pi
    
    # Try multiple initializations
    initializations = []
    initializations.append(_fibonacci_points())
    for seed in [42, 123, 999, 555, 1234, 5678, 9012, 3456]:
        initializations.append(_random_points(seed))
        
    for init_pts in initializations:
        if time.time() > _DEADLINE - 25:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Phase 1: Repulsion refinement to spread points
            result_rep = minimize(
                _repulsion_energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 800}
            )
            
            x1 = result_rep.x
            
            # Phase 1.5: Spherical Lloyd relaxation
            pts_lloyd = _spherical_lloyd_iteration(_to_sphere(x1.reshape(N, 3)), n_iters=5)
            x1 = pts_lloyd.flatten()
            
            # Phase 2: Multi-stage beta cascade for covering radius refinement
            # Start with low beta for smooth landscape, increase for sharpness
            beta_stages = [10.0, 50.0, 200.0]
            x_current = x1
            
            for beta in beta_stages:
                if time.time() > _DEADLINE - 15:
                    break
                    
                def _obj_with_beta(x_flat, _beta=beta):
                    return _smooth_objective_and_grad(x_flat, beta=_beta)
                
                result = minimize(
                    _obj_with_beta,
                    x_current,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1500}
                )
                
                x_current = result.x
                opt_pts = _to_sphere(x_current.reshape(N, 3))
                hull, radii, exact_obj = _compute_hull_and_radii(opt_pts)
                
                if exact_obj < best_obj:
                    best_obj = exact_obj
                    best_pts = opt_pts.copy()
                    
        except Exception as e:
            pass
            
    # Final high-precision polish with very high beta
    if best_pts is not None and time.time() < _DEADLINE - 5:
        x0 = best_pts.flatten()
        
        def _final_obj(x_flat):
            return _smooth_objective_and_grad(x_flat, beta=500.0)
        
        try:
            result = minimize(
                _final_obj,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 2000}
            )
            opt_pts = _to_sphere(result.x.reshape(N, 3))
            hull, radii, exact_obj = _compute_hull_and_radii(opt_pts)
            if exact_obj < best_obj:
                best_obj = exact_obj
                best_pts = opt_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_points()
        
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
