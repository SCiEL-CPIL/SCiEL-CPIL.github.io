"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    if n < 2:
        return 0.0
    # Vectorized distance calculation
    # Use squared distances to avoid sqrt where possible, but log needs d
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Avoid log(0) on diagonal
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    
    if np.any(d_upper < 1e-15):
        return -1e18
        
    # Sum log(1/d) = -sum(log(d))
    # Use fsum for precision
    return math.fsum(-np.log(d_upper).tolist())

def _log_energy_grad(pts):
    """
    Compute logarithmic energy and Riemannian gradient on the sphere.
    The gradient is in the tangent space at each point.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    # Gradient of E = sum_{i<j} -log(||x_i - x_j||)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: log(1/d) = -log(d). d/dx (-log(||x||)) = - (1/||x||^2) * x.
    # So for pair (i,j), force on i is (x_i - x_j) / d_ij^2.
    
    # Vectorized gradient calculation
    # inv_d2[i,j] = 1/d_ij^2
    inv_d2 = np.zeros_like(d2)
    mask_non_diag = ~np.eye(n, dtype=bool)
    inv_d2[mask_non_diag] = 1.0 / d2[mask_non_diag]
    
    # grad[i] = sum_j inv_d2[i,j] * (x_i - x_j)
    # diff = x_i - x_j
    grad = np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient onto tangent space: grad_tan = grad - <grad, x_i> x_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad

def _exp_map(pts, vecs):
    """
    Riemannian exponential map on the unit sphere.
    Maps tangent vectors vecs at pts to the manifold.
    exp_p(v) = p*cos(||v||) + (v/||v||)*sin(||v||)
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    
    cos_term = np.cos(safe_norms)
    sin_term = np.sin(safe_norms) / safe_norms
    
    # Handle zero norm vectors (stay at point)
    zero_mask = (norms < 1e-10).flatten()
    
    new_pts = pts * cos_term + vecs * sin_term
    
    # For zero norms, result should be pts
    new_pts[zero_mask] = pts[zero_mask]
    
    # Normalize to counter numerical drift
    final_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    final_norms[final_norms < 1e-10] = 1.0
    return new_pts / final_norms

def _log_map(pts, vecs):
    """
    Riemannian logarithmic map (inverse of exp map).
    Given p and y, find v in T_p S^2 such that exp_p(v) = y.
    v = (arccos(<p,y>)) / ||p-y|| * (y - <p,y>p)
    """
    dots = np.sum(pts * vecs, axis=1, keepdims=True)
    dots = np.clip(dots, -1.0, 1.0)
    theta = np.arccos(dots)
    
    # v direction is y - <p,y>p
    proj = vecs - dots * pts
    proj_norms = np.linalg.norm(proj, axis=1, keepdims=True)
    
    # If theta is close to 0 or pi, handle carefully
    safe_theta = np.where(np.abs(theta) < 1e-10, 1.0, theta)
    
    v = proj * (safe_theta / np.where(proj_norms < 1e-10, 1.0, proj_norms))
    
    # If theta is 0, v is 0
    zero_mask = (np.abs(theta) < 1e-10).flatten()
    v[zero_mask] = 0.0
    
    return v

def _optimize_riemannian_cg(pts0, deadline, max_iter=1000):
    """
    Riemannian Conjugate Gradient method.
    """
    pts = pts0.copy()
    e, grad = _log_energy_grad(pts)
    
    if e < -1e17:
        return pts, e
        
    d = -grad
    y_prev = None
    alpha = 0.01
    
    best_pts = pts.copy()
    best_e = e
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Line search on the manifold
        # We want to find t such that f(exp_p(t*d)) is minimized
        # Simple backtracking line search
        t = 1.0
        cur_e = e
        cur_grad = grad
        
        for _ in range(50):
            new_pts = _exp_map(pts, t * d)
            new_e, new_grad = _log_energy_grad(new_pts)
            
            if new_e < cur_e:
                pts = new_pts
                e = new_e
                grad = new_grad
                break
            else:
                t *= 0.5
                if t < 1e-8:
                    break
        else:
            # If line search fails, break
            break
            
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            
        # Check convergence
        if np.linalg.norm(grad) < 1e-10:
            break
            
        # Compute new search direction
        if y_prev is not None:
            # Fletcher-Reeves beta
            beta = np.sum(grad * grad) / np.sum(y_prev * y_prev)
            beta = max(0.0, beta)
            d = -grad + beta * d
        else:
            d = -grad
            
        # Project d onto tangent space at new pts
        radial = np.sum(d * pts, axis=1, keepdims=True)
        d -= radial * pts
        
        y_prev = grad - np.sum((grad - _log_energy_grad(pts)[1]) * 0) # Simplified, usually y = g_k - g_{k-1}
        # Proper R-CG uses y = g_k - g_{k-1} projected
        # For simplicity and robustness, we restart CG every 20 iterations or if not descent
        if it % 20 == 0:
            d = -grad
            radial = np.sum(d * pts, axis=1, keepdims=True)
            d -= radial * pts
            y_prev = None
        else:
             y_prev = grad - _log_energy_grad(pts)[1] # This is wrong, need previous grad stored properly
             # Let's use a simpler momentum-based Riemannian update if CG gets complex
             # Actually, let's switch to Riemannian L-BFGS or just Momentum which is easier to implement correctly without state bugs.
             
    return best_pts, best_e

def _optimize_riemannian_momentum(pts0, deadline, max_iter=2000):
    """
    Riemannian Gradient Descent with Momentum.
    This is more robust than CG for this specific task within the time limit.
    """
    pts = pts0.copy()
    v = np.zeros_like(pts)
    lr = 0.01
    momentum = 0.9
    
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        e, grad = _log_energy_grad(pts)
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            
        if np.linalg.norm(grad) < 1e-12:
            break
            
        # Update velocity in tangent space
        v = momentum * v - lr * grad
        
        # Project v to tangent space at current pts
        radial = np.sum(v * pts, axis=1, keepdims=True)
        v -= radial * pts
        
        # Geodesic step
        new_pts = _exp_map(pts, v)
        
        # Adaptive step size (simplified)
        new_e = _log_energy(new_pts)
        if new_e < e:
            pts = new_pts
            # Slightly increase lr if consistent improvement
            if it > 100:
                lr = min(lr * 1.005, 0.05)
        else:
            # Reduce step
            lr *= 0.5
            v *= 0.5
            
    return best_pts, best_e

def _optimize_lbfgs_riemannian(pts0, deadline):
    """
    Use scipy minimize but with a wrapper that handles the manifold.
    Since scipy works on R^N, we use the projection method but with tighter controls.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e
        
    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()
        
    try:
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _log_energy(best_pts)
            return best_pts, best_energy
        else:
            return pts0, _log_energy(pts0)
    except Exception:
        return pts0, _log_energy(pts0)

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _symmetry_aware_init(n, seed=0):
    """
    Generate points using Icosahedral symmetry groups to ensure good spread.
    """
    # Icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = []
    rng = np.random.RandomState(seed)
    
    # Start with the 12 vertices
    for v in verts:
        pts.append(v.copy())
        
    # Add points in orbits of 5 or 10 or 20
    # Simple approach: take random points and apply icosahedral rotation group
    # Rotation matrices for icosahedron are complex, so we just use random perturbations
    # of the vertices and their neighbors to fill the space
    
    remaining = n - len(pts)
    for i in range(remaining):
        # Pick a random vertex and perturb
        v = verts[rng.randint(0, 12)]
        # Add small random noise
        noise = rng.randn(3) * 0.2
        v = v + noise
        v /= np.linalg.norm(v)
        pts.append(v)
        
    return np.array(pts)

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pts = None
    best_energy = np.inf
    
    # Initialization strategies
    inits = []
    
    # 1. Symmetry-aware init
    inits.append(_symmetry_aware_init(N, seed=1))
    inits.append(_symmetry_aware_init(N, seed=2))
    
    # 2. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib_pts)
    
    # 3. Random
    rng = np.random.RandomState(42)
    rand_pts = rng.randn(N, 3)
    rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
    inits.append(rand_pts)
    
    # Optimization loop
    for init_pts in inits:
        if time.time() > deadline - 20:
            break
            
        # Phase 1: Riemannian Momentum (fast, robust)
        pts_mom, e_mom = _optimize_riemannian_momentum(init_pts, deadline - 10)
        
        if e_mom < best_energy:
            best_energy = e_mom
            best_pts = pts_mom
            
        # Phase 2: L-BFGS (polish)
        if time.time() < deadline - 5:
            pts_lbfgs, e_lbfgs = _optimize_lbfgs_riemannian(pts_mom, deadline - 2)
            if e_lbfgs < best_energy:
                best_energy = e_lbfgs
                best_pts = pts_lbfgs
                
        # Phase 3: Final Riemannian Momentum polish (small steps)
        if best_pts is not None and time.time() < deadline - 1:
             # Create a tighter optimizer by modifying lr in a wrapper? 
             # For now, just run momentum again with existing logic which adapts lr
             pts_final, e_final = _optimize_riemannian_momentum(best_pts, deadline)
             if e_final < best_energy:
                 best_energy = e_final
                 best_pts = pts_final

    return best_pts
# EVOLVE-BLOCK-END
