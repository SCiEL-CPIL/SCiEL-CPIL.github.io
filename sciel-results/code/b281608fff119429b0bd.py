"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_data(pts):
    """
    Compute convex hull and data needed for objective and gradient.
    Returns:
        facets: array of shape (F, 3) indices of vertices
        cos_radii: array of shape (F,) cosines of circumradii (distance from origin to facet plane)
        grad_cos: array of shape (F, N, 3) gradients of cos_radii w.r.t points
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
    
    facets = hull.simplices
    F = len(facets)
    if F == 0:
        return None, None, None
        
    cos_radii = np.zeros(F)
    grad_cos = np.zeros((F, N, 3))
    
    for i, simplex in enumerate(facets):
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-12:
            continue
            
        # Distance from origin to plane
        d = np.dot(n, v1) / n_norm
        
        # Ensure outward normal (d > 0)
        if d < 0:
            d = -d
            n = -n
            
        cos_radii[i] = d
        
        # Gradients of d = dot(n, v1) / ||n||
        # d = (n · v1) / ||n||
        # Let num = n · v1, den = ||n||
        # dd/du = (den * d(num)/du - num * d(den)/du) / den^2
        # d(num)/du = d(n·v1)/du
        # d(den)/du = d(||n||)/du = (n · dn/du) / ||n||
        
        # dn/du:
        # n = (v2-v1) x (v3-v1)
        # dn/dv1 = -e2 x e1 - e1 x e2 = 2 * e2 x e1  (Note: cross(a,b) = -cross(b,a))
        # Wait, d(v2-v1)/dv1 = -I, d(v3-v1)/dv1 = -I
        # d((v2-v1) x (v3-v1))/dv1 = (-I)(v3-v1) x e1 + (v2-v1) x (-I) = -e2 x e1 - e1 x e2 = -e2 x e1 + e2 x e1 = 0?
        # No. Product rule for cross product: d(u x v) = du x v + u x dv.
        # u = v2-v1, v = v3-v1.
        # wrt v1: du = -dx, dv = -dx.
        # d n / dv1 = (-dx) x e2 + e1 x (-dx) = -dx x e2 - e1 x dx = e2 x dx - e1 x dx = (e2 - e1) x dx.
        # Let's check with vectors.
        # n = e1 x e2.
        # dn/dv1:
        # e1 depends on v1, e2 depends on v1.
        # de1/dv1 = -I. de2/dv1 = -I.
        # dn/dv1 = (-I) x e2 + e1 x (-I).
        # Matrix form: For vector x, (-I) x e2 is the matrix such that M x = -x x e2 = e2 x x.
        # So dn/dv1 is the linear map x -> e2 x x - e1 x x = (e2 - e1) x x.
        
        # wrt v2:
        # de1/dv2 = I. de2/dv2 = 0.
        # dn/dv2 = I x e2 + e1 x 0 = x x e2 = -e2 x x.
        
        # wrt v3:
        # de1/dv3 = 0. de2/dv3 = I.
        # dn/dv3 = 0 x e2 + e1 x I = e1 x x.
        
        # Let's define matrices A1, A2, A3 such that dn/dvj = Aj @ dx.
        # A1: x -> (e2 - e1) x x. Note (a x x) = - (x x a) = - (a x)^T x? No.
        # Cross product matrix [a]_x such that [a]_x x = a x x.
        # So A1 = [e2-e1]_x.
        # A2: x -> -e2 x x = -[e2]_x x. So A2 = -[e2]_x.
        # A3: x -> e1 x x = [e1]_x x. So A3 = [e1]_x.
        
        def cross_mat(a):
            return np.array([
                [0, -a[2], a[1]],
                [a[2], 0, -a[0]],
                [-a[1], a[0], 0]
            ])
            
        A1 = cross_mat(e2 - e1)
        A2 = -cross_mat(e2)
        A3 = cross_mat(e1)
        
        # Gradients of num = n · v1
        # d(num)/dv1 = n + (dn/dv1)^T n = n + A1^T n
        # d(num)/dv2 = (dn/dv2)^T n = A2^T n
        # d(num)/dv3 = (dn/dv3)^T n = A3^T n
        
        grad_num_1 = n + A1.T @ n
        grad_num_2 = A2.T @ n
        grad_num_3 = A3.T @ n
        
        # Gradients of den = ||n||
        # d(den)/dvj = (n · (dn/dvj)) / ||n|| = (n · (Aj @ dx)) / ||n|| = ((Aj^T n) · dx) / ||n||
        # So grad_den_j = Aj^T n / n_norm
        
        grad_den_1 = A1.T @ n / n_norm
        grad_den_2 = A2.T @ n / n_norm
        grad_den_3 = A3.T @ n / n_norm
        
        # dd/du = (den * grad_num - num * grad_den) / den^2
        # num = n · v1 = d * n_norm
        # den = n_norm
        # dd/du = (n_norm * grad_num - (d * n_norm) * grad_den) / n_norm^2
        #       = (grad_num - d * grad_den) / n_norm
        
        factor = 1.0 / n_norm
        
        grad_d_1 = (grad_num_1 - d * grad_den_1) * factor
        grad_d_2 = (grad_num_2 - d * grad_den_2) * factor
        grad_d_3 = (grad_num_3 - d * grad_den_3) * factor
        
        grad_cos[i, simplex[0], :] = grad_d_1
        grad_cos[i, simplex[1], :] = grad_d_2
        grad_cos[i, simplex[2], :] = grad_d_3
        
    return facets, cos_radii, grad_cos

def _objective_and_grad(x_flat):
    """
    Objective: log-sum-exp(-beta * cos_radii)
    This is a smooth approximation to -min(cos_radii) / beta.
    Minimizing this maximizes the minimum cosine (minimizes covering radius).
    """
    pts = x_flat.reshape(N, 3)
    
    # Normalize to sphere (soft projection for gradient flow? No, hard projection in eval, but for gradient we assume manifold)
    # Actually, for gradient descent on sphere, we should project gradient to tangent space.
    # But simpler: just use standard gradient and project steps back to sphere.
    
    facets, cos_radii, grad_cos = _hull_data(pts)
    
    if facets is None:
        # Invalid hull, return large penalty
        return 1e10, np.zeros_like(x_flat)
        
    # Clip cosines to avoid numerical issues
    cos_radii = np.clip(cos_radii, -1.0, 1.0)
    
    beta = 50.0 # Temperature for log-sum-exp
    
    # Soft max of -cos_radii
    # We want to minimize max angle -> maximize min cos.
    # Equivalent to minimize -min(cos).
    # Surrogate: (1/beta) log(sum(exp(-beta * cos_i)))
    
    # Numerical stability
    max_neg_cos = np.max(-cos_radii)
    exp_terms = np.exp(-beta * cos_radii - beta * max_neg_cos)
    log_sum_exp = np.log(np.sum(exp_terms)) + beta * max_neg_cos
    obj = log_sum_exp / beta
    
    # Gradient of obj w.r.t x
    # d/dx [ (1/beta) log(sum(exp(-beta * c_i))) ]
    # = (1/beta) * [ sum(-beta * exp(-beta * c_i) * dc_i/dx) / sum(exp(-beta * c_i)) ]
    # = - [ sum(exp(-beta * c_i) * dc_i/dx) / sum(exp(-beta * c_i)) ]
    # = - [ sum(w_i * dc_i/dx) ] where w_i = exp(-beta * c_i) / sum(exp(-beta * c_i))
    
    weights = exp_terms / np.sum(exp_terms)
    
    # grad_obj = - sum_i w_i * grad_cos[i]
    # grad_cos shape (F, N, 3)
    # weights shape (F,)
    # Result shape (N, 3)
    grad_obj = -np.einsum('f,fnk->nk', weights, grad_cos)
    
    # Project gradient to tangent space of sphere
    # For each point p_i, grad_tangent = grad - (grad . p_i) p_i
    pts_norm = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    grad_flat = grad_obj.flatten()
    grad_tan_flat = np.zeros_like(grad_flat)
    
    for i in range(N):
        p = pts_norm[i]
        g = grad_obj[i]
        # Project g to tangent space at p
        g_tan = g - np.dot(g, p) * p
        grad_tan_flat[3*i:3*i+3] = g_tan
        
    return obj, grad_tan_flat

def construct():
    start_time = time.time()
    max_time = 145.0
    
    # Initial guess: Fibonacci spiral
    pts = _initial_fib()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Lloyd's relaxation for initial improvement (fast, no gradients needed)
    # Simple version: move points towards centroid of their Voronoi cell? 
    # Approximation: move towards average of neighbors? 
    # Let's just do a few steps of "move towards worst facet center" to get out of bad local minima quickly
    for _ in range(50):
        if time.time() - start_time > max_time - 30: break
        facets, cos_radii, _ = _hull_data(pts)
        if facets is None: continue
        worst_idx = np.argmin(cos_radii)
        simplex = facets[worst_idx]
        # Get circumcenter direction
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        e1, e2 = v2-v1, v3-v1
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        d = np.dot(n, v1)
        if d < 0: n = -n
        
        # Move vertices towards n
        for idx in simplex:
            pts[idx] += 0.05 * n
            pts[idx] /= np.linalg.norm(pts[idx])
            
    x0 = pts.flatten()
    
    # Optimization
    best_x = x0.copy()
    best_val = _objective_and_grad(x0)[0]
    
    # Use L-BFGS-B or BFGS? BFGS is usually good for unconstrained (with projection).
    # Since we project gradient, we can use standard BFGS on the manifold if we project steps.
    # SciPy minimize doesn't handle manifold projection of steps automatically.
    # We will implement a simple gradient descent with line search and projection.
    
    lr = 0.01
    for it in range(1000):
        if time.time() - start_time > max_time - 10:
            break
            
        val, grad = _objective_and_grad(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
            
        # Line search
        alpha = lr
        improved = False
        for _ in range(10):
            new_x = x0 + alpha * grad
            # Project to sphere
            new_pts = new_x.reshape(N, 3)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_val, _ = _objective_and_grad(new_pts.flatten())
            
            if new_val < val:
                x0 = new_pts.flatten()
                improved = True
                break
            alpha *= 0.5
            
        if not improved:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-6:
                break
                
    # Final check and polish
    final_pts = best_x.reshape(N, 3)
    final_pts /= np.linalg.norm(final_pts, axis=1, keepdims=True)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(final_pts[i] - final_pts[j]) < 1e-10:
                final_pts[j] += np.random.randn(3) * 0.001
                final_pts[j] /= np.linalg.norm(final_pts[j])
                
    return final_pts
# EVOLVE-BLOCK-END
