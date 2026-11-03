"""
Combined program: Uses grid-based surrogate for fast gradients and 
ConvexHull for exact evaluation.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _exact_covering_radius(points):
    """
    Compute the exact covering radius as the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        points = np.asarray(points, dtype=float)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        points = points / norms
        
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            o = n / norm_n
            centroid = (p1 + p2 + p3) / 3.0
            if np.dot(o, centroid) < 0:
                o = -o
            cos_r = np.dot(o, p1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        return np.pi

def _surrogate_loss_and_grad(v, beta, Y):
    """
    Compute the smooth surrogate for the covering radius and its gradient.
    Uses soft-min for distance to nearest point, soft-max for max over sphere.
    v: flattened array of N points (3N values)
    beta: sharpness parameter
    Y: grid of M points on the sphere
    """
    X = v.reshape(-1, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    X_normed = X / safe_norms
    
    # Compute angular distances D (M x N)
    # dots = Y @ X.T
    dots = Y @ X_normed.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Softmin over sites j for each grid point i
    # v_i = -1/beta * log(sum_j exp(-beta * dists_ij))
    min_dists = np.min(dists, axis=1, keepdims=True)
    exp_terms = np.exp(-beta * (dists - min_dists))
    sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
    # v_soft is M x 1
    v_soft = -(min_dists + np.log(sum_exp)) / beta
    
    # Softmax over grid points i
    # F = 1/beta * log(sum_i exp(beta * v_i))
    max_v = np.max(v_soft.flatten())
    exp_v = np.exp(beta * (v_soft.flatten() - max_v))
    sum_exp_v = np.sum(exp_v)
    F = (max_v + np.log(sum_exp_v)) / beta
    
    # Weights p_i (M)
    p = exp_v / sum_exp_v
    
    # Weights q_ij (M x N)
    q = exp_terms / sum_exp
    
    # Gradient calculation
    # We want grad_v such that dF = sum grad_v . dv
    # The constraint is ||X_j||=1.
    # The gradient with respect to X_j in the ambient space is:
    # grad_X_j = sum_i p_i * q_ij * grad_{X_j} dists_ij
    # grad_{X_j} arccos(Y_i . X_j) = (X_j (Y_i . X_j) - Y_i) / sin(dists_ij)
    
    sin_dists = np.sqrt(1.0 - dots**2)
    sin_dists[sin_dists < 1e-8] = 1e-8
    
    # Vectorized gradient computation
    # term_ij = (X_j * dots_ij - Y_i) / sin_dists_ij
    # We need to sum over i: sum_i (p_i * q_ij * term_ij)
    
    # Shapes:
    # dots: M, N
    # sin_dists: M, N
    # X_normed: N, 3
    # Y: M, 3
    
    # To avoid huge memory usage for M*N*3, we can do it in chunks or use einsum carefully.
    # M=20000, N=50 -> 1M entries. M*N*3 = 3M floats = 24MB. This is acceptable.
    
    # term[i, j, k] = (X_normed[j, k] * dots[i, j] - Y[i, k]) / sin_dists[i, j]
    # This is (M, N, 3)
    
    # Let's compute weights first: w[i, j] = p[i] * q[i, j]
    w = p[:, np.newaxis] * q  # M, N
    
    # Now, grad_X[j, k] = sum_i w[i, j] * (X_normed[j, k] * dots[i, j] - Y[i, k]) / sin_dists[i, j]
    # = X_normed[j, k] * sum_i (w[i, j] * dots[i, j] / sin_dists[i, j]) - sum_i (w[i, j] * Y[i, k] / sin_dists[i, j])
    
    # Let A[i, j] = w[i, j] / sin_dists[i, j]
    A = w / sin_dists  # M, N
    
    # Term 1: sum_i A[i, j] * dots[i, j]  -> (N,)
    term1 = np.sum(A * dots, axis=0)  # N
    
    # Term 2: sum_i A[i, j] * Y[i, k] -> (N, 3)
    # A is (M, N), Y is (M, 3).
    # We want sum over i of A[i, j] * Y[i, k] for each j, k.
    # This is A.T @ Y ? No.
    # A.T is (N, M). Y is (M, 3). Result is (N, 3).
    term2 = A.T @ Y  # N, 3
    
    grad_X = X_normed * term1[:, np.newaxis] - term2  # N, 3
    
    # Project gradient to tangent space?
    # The L-BFGS-B optimizes over the unconstrained space R^{3N}.
    # The function F is defined on the sphere.
    # If we use X_normed in the calculation, the gradient with respect to the raw v is:
    # grad_v = grad_X / norms.flatten() ?
    # Actually, if we treat X as a variable on the sphere, the gradient is tangent.
    # But L-BFGS-B doesn't know the manifold.
    # Standard trick: If we normalize X inside the function, the gradient with respect to the raw input v is:
    # grad_v = (grad_X - (grad_X . X_normed) X_normed) / ||v||
    # Wait, let's verify.
    # Let X = v / ||v||.
    # dF = grad_X . dX
    # dX = (dv - (v . dv) v/||v||^2) / ||v|| = (dv - (v . dv) X) / ||v||
    # dF = grad_X . (dv - (v . dv) X) / ||v||
    #     = (grad_X / ||v||) . dv - (grad_X . X / ||v||) (v . dv)
    #     = (grad_X / ||v||) . dv - (grad_X . X / ||v||) (||v|| X . dv)
    #     = (grad_X / ||v||) . dv - (grad_X . X) X . dv
    #     = [ grad_X / ||v|| - (grad_X . X) X ] . dv
    # So grad_v = grad_X / ||v|| - (grad_X . X) X
    # Note: grad_X is already tangent? No, grad_X computed above is the gradient in the ambient space of the normalized points.
    # Actually, the formula for gradient of f(X) where X is constrained to sphere, expressed in ambient coords, is the projection of the unconstrained gradient.
    # Here, we computed grad_X as if X were free, but using X_normed in the function.
    # The term `grad_X` calculated as `X_normed * term1 - term2` is the gradient of the function with respect to the *normalized* coordinates?
    # No, it's the gradient of the scalar field defined on R^3 restricted to the sphere.
    # The correct gradient for the optimizer (which varies v) is:
    # grad_v = (grad_X - (grad_X . X_normed) X_normed) / ||v||
    
    grad_X_tangent = grad_X - np.sum(grad_X * X_normed, axis=1, keepdims=True) * X_normed
    grad_v = grad_X_tangent / safe_norms.flatten()
    
    return F, grad_v.flatten()

def _generate_grid(M):
    """Generate a dense grid of points on the sphere."""
    i_grid = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.maximum(0.0, 1.0 - z_grid * z_grid))
    Y = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    return Y

def construct():
    global START_TIME, TIME_LIMIT
    
    # Generate a dense grid for the surrogate
    # 20000 points is a good balance. 
    M = 20000
    Y = _generate_grid(M)
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    configs.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 2. Random configurations
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        configs.append(pts)
        
    # 3. Jittered Fibonacci
    base = configs[0].copy()
    jitter = np.random.randn(N, 3) * 0.1
    base += jitter
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    configs.append(base)
    
    candidates = []
    
    # Optimization stages
    # Start with lower beta for broad exploration, increase for refinement
    betas = [20, 50, 100, 200, 400]
    
    for start_idx, init_pts in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
            
        v = init_pts.flatten()
        current_points = init_pts.copy()
        
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            try:
                res = minimize(
                    _surrogate_loss_and_grad,
                    v,
                    args=(beta, Y),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                v = res.x
                current_points = v.reshape(-1, 3)
                current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            except Exception:
                pass
        
        # Evaluate exact radius for this candidate
        rad = _exact_covering_radius(current_points)
        candidates.append((rad, current_points))
        
        if rad < best_radius:
            best_radius = rad
            best_points = current_points.copy()
            
    # Polish the best candidate with higher beta
    if best_points is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        v = best_points.flatten()
        for beta in [800, 1200, 2000]:
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
            try:
                res = minimize(
                    _surrogate_loss_and_grad,
                    v,
                    args=(beta, Y),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                v = res.x
                best_points = v.reshape(-1, 3)
                best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
            except Exception:
                break
        
        # Final check
        final_rad = _exact_covering_radius(best_points)
        if final_rad < best_radius:
            best_radius = final_rad
            best_points = best_points  # Already updated

    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_points
