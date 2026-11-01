import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 130
_EPS = 1e-12
DEADLINE = 150.0

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_grid(n_theta=50, n_phi=100):
    """Generate a dense spherical grid for coverage evaluation."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and angular radii of all facets.
    Returns max radius or None if failed.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: arccos(normal . vertex)
        # Use A, B, C for robustness? No, normal . any vertex is same if equidistant from circumcenter?
        # Actually, for a facet on the unit sphere, the circumcenter is the normal direction.
        # The distance from center (origin) to facet plane is 1 (since points are on unit sphere).
        # Wait, the "covering radius" is the max distance from any point on sphere to nearest config point.
        # This is equivalent to the max circumradius of the spherical Voronoi cells.
        # The circumradius of a spherical triangle with vertices A, B, C on unit sphere is the angle
        # between the normal vector (circumcenter) and any of the vertices.
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    # Generate grid
    grid = _generate_grid(40, 80) # 3200 points, reasonable for speed
    n_grid = len(grid)
    
    # Initial points
    pts = _initial_fib()
    
    # Precompute grid norms (should be 1)
    grid_norms = np.linalg.norm(grid, axis=1, keepdims=True)
    grid /= grid_norms
    
    # RBF bandwidth
    # We want the kernel to be narrow enough to approximate distance but smooth.
    # Sigma ~ 0.1 rad?
    sigma = 0.05
    
    def objective_and_grad(x_flat):
        if time.time() > deadline:
            return 0.0, np.zeros_like(x_flat)
            
        pts_x = x_flat.reshape(N, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(pts_x, axis=1, keepdims=True)
        pts_x = pts_x / (norms + _EPS)
        
        # Compute coverage field: sum of RBFs
        # d^2 = 2 - 2 * (grid . pts_x^T)
        # K = exp(-d^2 / (2 * sigma^2))
        
        # Vectorized computation
        # grid: (M, 3), pts_x: (N, 3)
        # dot: (M, N)
        dot = grid @ pts_x.T
        d2 = 2.0 - 2.0 * dot
        d2 = np.maximum(d2, 0.0)
        
        exponent = -d2 / (2.0 * sigma * sigma)
        K = np.exp(exponent)
        
        # Coverage at each grid point is sum of K over all N points
        coverage = np.sum(K, axis=1)
        
        # Objective: smooth max (log-sum-exp)
        # We want to minimize the max coverage? No, we want to maximize the min distance.
        # Wait, the task is to MINIMIZE the covering radius.
        # The covering radius is the max over y of min_i d(y, p_i).
        # If we use RBFs, a high value of K means "close".
        # So min_i d(y, p_i) corresponds to max_i K(y, p_i).
        # So covering radius corresponds to min_y max_i K(y, p_i).
        # We want to MINIMIZE the covering radius, which means we want to MINIMIZE the value
        # that represents the "worst" coverage.
        # Wait.
        # If points are well distributed, the "gaps" are small.
        # In the gaps, the max RBF value is small.
        # So we want to MAXIMIZE the minimum of the max RBF value?
        # Let's check.
        # If points are clustered, some regions have max RBF ~ 0.
        # If points are spread, all regions have max RBF ~ 1 (or close).
        # So we want to MAXIMIZE min_y (max_i K(y, p_i)).
        # This is a max-min problem.
        # To use gradient descent (minimization), we minimize -min_y (max_i K(y, p_i)).
        # Or we can approximate max_i K with logsumexp of K, and min_y with negative logsumexp of -coverage?
        # Let's stick to the geometric definition: minimize the max angular distance.
        # Surrogate: Minimize the "gap" size.
        # Let's use the property that the covering radius is the max facet circumradius.
        # But we want a smooth function.
        
        # Alternative: Minimize the sum of exp(alpha * d(y, nearest)) ?
        # Let's use the RBF sum as a "potential".
        # If we maximize the minimum potential, we fill the gaps.
        # Objective: - min(coverage)
        # Smooth min: - logsumexp(-coverage / temp)
        
        temp = 0.01
        obj = -logsumexp(-coverage / temp)
        
        # Gradient
        # d obj / d x = - d/dx ( - logsumexp(-coverage/temp) )
        # = d/dx ( logsumexp(-coverage/temp) )
        # Let z = -coverage / temp
        # logsumexp(z) = log(sum(exp(z)))
        # d logsumexp(z) / dz = exp(z) / sum(exp(z)) = softmax(z)
        # dz / d coverage = -1 / temp
        # d coverage / d x = sum_i dK_i / dx
        # K_i = exp(-d2_i / (2 sigma^2))
        # dK_i / dx = K_i * (1/sigma^2) * (x - p_i)  (chain rule: d(-d2)/dx = 2(x-p_i), d2 = |x-p_i|^2 = 2-2x.p_i, d(d2)/dx = -2p_i? No.
        # d2 = (x-p_i)^2 = x^2 - 2x.p_i + p_i^2 = 2 - 2x.p_i (since x,p_i on unit sphere)
        # d(d2)/dx = -2 p_i
        # dK_i/dx = K_i * (-1/(2 sigma^2)) * (-2 p_i) = K_i / sigma^2 * p_i
        # Wait, x is the grid point (fixed), p_i is the variable.
        # dK_i/dp_i = K_i / sigma^2 * (x - p_i) ?
        # Let's re-derive.
        # K = exp( (2 x . p_i - 2) / (2 sigma^2) ) = exp( (x . p_i - 1) / sigma^2 )
        # dK/dp_i = K * (1/sigma^2) * x
        # So d coverage / d p_i = sum_j K_j * (1/sigma^2) * x_j  (for the j-th point in sum? No, sum over i)
        # coverage(y) = sum_i K(y, p_i)
        # d coverage(y) / d p_k = K(y, p_k) * (1/sigma^2) * y
        
        # So grad_obj w.r.t p_k:
        # sum_y softmax(z_y) * (-1/temp) * d coverage(y) / d p_k
        # = sum_y softmax(z_y) * (-1/temp) * K(y, p_k) / sigma^2 * y
        
        # Let w_y = softmax(z_y) * (-1/temp)
        # grad_k = (1/sigma^2) * sum_y w_y * K(y, p_k) * y
        
        # Compute weights
        z = -coverage / temp
        z -= np.max(z) # for stability
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        w = exp_z / sum_exp_z * (-1.0 / temp)
        
        # w shape: (M,)
        # K shape: (M, N)
        # We need sum_y w_y * K(y, p_k) * y for each k.
        # weighted_K = w[:, None] * K  (M, N)
        # grad = (1/sigma^2) * (weighted_K.T @ grid)  (N, 3)
        
        weighted_K = w[:, None] * K
        grad = (1.0 / (sigma * sigma)) * (weighted_K.T @ grid)
        
        # grad is d obj / d p_k (tangent space? No, Euclidean)
        # We need to project gradient to tangent space?
        # L-BFGS-B works in Euclidean space.
        # The constraint is |p_k| = 1.
        # If we just update p_k -= lr * grad, we leave the sphere.
        # But we normalize in the objective function.
        # This is a common trick: optimize in R^3, normalize inside objective.
        # The gradient of the normalized objective is not simply the gradient of the unnormalized one.
        # However, for small steps, it's often sufficient.
        # Better: project gradient onto tangent space.
        # grad_tangent = grad - (grad . p_k) p_k
        
        pts_norm = np.linalg.norm(pts_x, axis=1, keepdims=True)
        pts_unit = pts_x / pts_norm
        
        grad_proj = grad - np.sum(grad * pts_unit, axis=1, keepdims=True) * pts_unit
        
        # Flatten gradient
        grad_flat = grad_proj.ravel()
        
        return obj, grad_flat

    # Optimize
    best_obj = np.inf
    best_x = pts.ravel()
    
    x0 = pts.ravel()
    
    # Use L-BFGS-B
    # We might need multiple restarts or careful step control.
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Polish with Hull
    # Convert back to points
    pts_opt = best_x.reshape(N, 3)
    pts_opt /= np.linalg.norm(pts_opt, axis=1, keepdims=True)
    
    # Local polish: move worst facet vertices towards pole
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
                
            # Compute hull
            try:
                hull = ConvexHull(pts_opt)
                faces = hull.simplices
                A = pts_opt[faces[:, 0]]
                B = pts_opt[faces[:, 1]]
                C = pts_opt[faces[:, 2]]
                
                AB = B - A
                AC = C - A
                cross = np.cross(AB, AC)
                area = 0.5 * np.linalg.norm(cross, axis=1)
                valid = area > _EPS
                if not np.any(valid):
                    break
                    
                normals = cross[valid]
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                centroids = (A[valid] + B[valid] + C[valid]) / 3.0
                signs = np.sum(normals * centroids, axis=1, keepdims=True)
                normals = normals * np.sign(signs)
                normals = normals / (norms + _EPS)
                
                cos_theta = np.sum(normals * A[valid], axis=1)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                ang_radii = np.arccos(cos_theta)
                
                worst_idx = np.argmax(ang_radii)
                v_indices = faces[valid][worst_idx]
                pole = normals[worst_idx]
                
                # Move vertices
                step = 0.01
                for v in v_indices:
                    dir_vec = pole - pts_opt[v]
                    dir_norm = np.linalg.norm(dir_vec)
                    if dir_norm > _EPS:
                        dir_vec /= dir_norm
                        pts_opt[v] += step * dir_vec
                        pts_opt[v] /= np.linalg.norm(pts_opt[v])
                        
            except Exception:
                break

    # Final normalization
    best_pts = np.array(best_x.reshape(N, 3), dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts /= (norms + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
