"""Rewritten program using Riemannian optimization with multiple strategies."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere (or normalized inside).
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
        u: (N, 3) normalized points
    """
    # Ensure points are normalized for calculation
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    u = pts / norms
    
    # Pairwise differences
    diff = u[:, None, :] - u[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Clip to avoid log(0) and division by 0
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy calculation
    # E = sum_{i<j} log(1/d) = -0.25 * sum_{i!=j} log(d_sq_ij)
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient calculation
    # dE/du_i = sum_{j!=i} (u_j - u_i) / ||u_i - u_j||^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    w_sum = np.sum(inv_dist_sq, axis=1)
    
    term1 = inv_dist_sq @ u
    term2 = u * w_sum[:, None]
    
    grad_raw = term1 - term2
    
    # Project to tangent space
    dot = np.sum(grad_raw * u, axis=1)
    grad_tan = grad_raw - dot[:, None] * u
    
    return energy, grad_tan, u

def _objective_and_grad(x_flat):
    """
    Objective function and gradient for scipy.minimize.
    x_flat: (3N,) vector of unnormalized coordinates.
    """
    x = x_flat.reshape(N, 3)
    
    energy, grad_tan, u = _energy_and_tangent_grad(x)
    
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    
    grad_x = grad_tan / norms
    
    return energy, grad_x.flatten()

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_random(rng):
    """Random points on the unit sphere."""
    pts = rng.normal(size=(N, 3))
    return _project_to_sphere(pts)

def _initial_antipodal(rng):
    """Antipodal symmetric points."""
    pts_half = rng.normal(size=(N//2, 3))
    pts_full = np.vstack([pts_half, -pts_half])
    return _project_to_sphere(pts_full)

def _optimize_lbfgs(start_pts, deadline, is_polish=False, maxiter=None):
    """
    Optimize using L-BFGS-B with time-based callback.
    """
    x0 = start_pts.flatten()
    
    if is_polish:
        ftol = 1e-16
        gtol = 1e-15
        if maxiter is None:
            maxiter = 5000
    else:
        ftol = 1e-12
        gtol = 1e-10
        if maxiter is None:
            maxiter = 3000
    
    class TimeOutCallback:
        def __init__(self, deadline):
            self.deadline = deadline
            self.timeout = False
            self.last_time = time.time()
            
        def __call__(self, xk, f, g):
            if time.time() > self.deadline:
                self.timeout = True
            return self.timeout
    
    cb = TimeOutCallback(deadline)
    
    options = {
        'maxiter': maxiter,
        'maxfun': 10 * maxiter,
        'ftol': ftol,
        'gtol': gtol,
        'iprint': 0
    }
    
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            callback=cb,
            options=options
        )
        return res.x.reshape(N, 3)
    except Exception:
        return start_pts.copy()

def _local_search(pts, deadline, max_iter=2000, step_scale=1.0):
    """
    Simple gradient descent with adaptive step size in tangent space.
    Used as a final polish or to escape shallow local minima.
    """
    pts = _project_to_sphere(pts.copy())
    e_curr, grad_tan, u = _energy_and_tangent_grad(pts)
    
    # Estimate initial step size based on gradient magnitude
    g_norm = np.linalg.norm(grad_tan)
    if g_norm < 1e-15:
        return pts, e_curr
    
    # Initial step: aim for ~1% change in energy per step initially
    step = 0.1 * step_scale
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Move in negative gradient direction (tangent)
        g_norm = np.linalg.norm(grad_tan)
        if g_norm < 1e-15:
            break
        grad_norm = grad_tan / g_norm
        
        # Try step
        new_pts = pts - step * grad_norm
        new_pts = _project_to_sphere(new_pts)
        
        e_new, grad_new, u_new = _energy_and_tangent_grad(new_pts)
        
        if e_new < e_curr:
            pts = new_pts
            e_curr = e_new
            grad_tan = grad_new
        else:
            # Decrease step
            step *= 0.5
            if step < 1e-12:
                break
        
    return pts, e_curr

def _riemannian_lbfgs(pts, deadline, maxiter=2000):
    """
    Riemannian L-BFGS optimization on the sphere manifold.
    Uses exponential map for retraction and logarithmic map for tangent space.
    """
    pts = _project_to_sphere(pts.copy())
    
    # Store recent gradients and step directions for L-BFGS
    m = 10  # memory size
    W = []  # list of (s, y) pairs
    rho = []  # 1/(y^T s)
    
    e_curr, g_tan, _ = _energy_and_tangent_grad(pts)
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-14:
            break
        
        # L-BFGS two-loop recursion to compute search direction
        # Start with r = g
        r = g_tan.copy()
        
        # First loop (backward)
        alpha = np.zeros(m)
        for i in range(len(W) - 1, -1, -1):
            s_i, y_i = W[i]
            rho_i = rho[i]
            alpha[i] = rho_i * np.sum(s_i * r)
            r = r - alpha[i] * y_i
        
        # Scale by gamma
        if np.sum(y_i * y_i) > 1e-16:
            gamma = np.sum(s_i * y_i) / np.sum(y_i * y_i)
        else:
            gamma = 1.0
        q = gamma * r
        
        # Second loop (forward)
        d = q.copy()
        for i in range(len(W)):
            s_i, y_i = W[i]
            rho_i = rho[i]
            beta = rho_i * np.sum(y_i * d)
            d = d + (alpha[i] - beta) * s_i
        
        # d is the search direction in tangent space
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-16:
            d = -g_tan
            d_norm = np.linalg.norm(d)
        
        # Line search with backtracking
        step = 1.0
        for ls_iter in range(20):
            # Exponential map: move along direction d by step
            # For small steps, exp_map(x, v) ≈ x + v (first order)
            # More accurately: exp_map uses rotation by angle ||v|| around axis v/||v||
            v = step * d
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-16:
                new_pts = pts.copy()
            else:
                # Exponential map on sphere
                angle = v_norm
                axis = v / angle
                new_pts = np.zeros_like(pts)
                for i in range(N):
                    # Rotate pts[i] by angle around axis
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    v_i = axis
                    dot_v = np.sum(pts[i] * v_i)
                    cross_v = np.cross(v_i, pts[i])
                    new_pts[i] = cos_a * pts[i] + sin_a * cross_v + (1 - cos_a) * dot_v * v_i
            
            new_pts = _project_to_sphere(new_pts)
            e_new, g_new, _ = _energy_and_tangent_grad(new_pts)
            
            # Armijo condition
            if e_new < e_curr + 1e-4 * step * np.sum(g_tan * d):
                # Accept step
                s = new_pts - pts  # Approximate step
                y = g_new - g_tan  # Approximate gradient change
                
                # Update L-BFGS memory
                sy = np.sum(s * y)
                if sy > 1e-12:
                    W.append((s, y))
                    rho.append(1.0 / sy)
                    if len(W) > m:
                        W.pop(0)
                        rho.pop(0)
                
                pts = new_pts
                e_curr = e_new
                g_tan = g_new
                break
            else:
                step *= 0.5
        
        else:
            # Line search failed, try smaller step
            step = 0.01
            v = step * d
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-16:
                angle = v_norm
                axis = v / angle
                new_pts = np.zeros_like(pts)
                for i in range(N):
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    v_i = axis
                    dot_v = np.sum(pts[i] * v_i)
                    cross_v = np.cross(v_i, pts[i])
                    new_pts[i] = cos_a * pts[i] + sin_a * cross_v + (1 - cos_a) * dot_v * v_i
                new_pts = _project_to_sphere(new_pts)
                pts = new_pts
                e_curr, g_tan, _ = _energy_and_tangent_grad(pts)
    
    return pts

def _conjugate_gradient(pts, deadline, max_iter=1000):
    """
    Riemannian conjugate gradient method on the sphere.
    """
    pts = _project_to_sphere(pts.copy())
    e_curr, g_tan, _ = _energy_and_tangent_grad(pts)
    
    d = -g_tan.copy()
    beta = 0.0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-14:
            break
        
        # Line search
        step = 1.0
        found = False
        for ls_iter in range(20):
            v = step * d
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-16:
                new_pts = pts.copy()
            else:
                angle = v_norm
                axis = v / angle
                new_pts = np.zeros_like(pts)
                for i in range(N):
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    v_i = axis
                    dot_v = np.sum(pts[i] * v_i)
                    cross_v = np.cross(v_i, pts[i])
                    new_pts[i] = cos_a * pts[i] + sin_a * cross_v + (1 - cos_a) * dot_v * v_i
            
            new_pts = _project_to_sphere(new_pts)
            e_new, g_new, _ = _energy_and_tangent_grad(new_pts)
            
            if e_new < e_curr:
                # Update
                pts = new_pts
                e_curr = e_new
                g_tan = g_new
                found = True
                break
            else:
                step *= 0.5
        
        if not found:
            break
        
        # Update conjugate direction
        g_new_norm_sq = np.sum(g_tan * g_tan)
        if g_norm > 1e-16:
            beta = g_new_norm_sq / (g_norm * g_norm)
        
        d = -g_tan + beta * d
        g_norm = np.linalg.norm(g_tan)
    
    return pts

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # Fibonacci
    inits.append(_initial_fibonacci())
    
    # Random
    for _ in range(8):
        inits.append(_initial_random(rng))
        
    # Antipodal
    for _ in range(4):
        inits.append(_initial_antipodal(rng))
        
    # Jittered Fibonacci
    fib = _initial_fibonacci()
    for _ in range(4):
        pts = fib + rng.normal(size=(N, 3)) * 0.5
        inits.append(_project_to_sphere(pts))
        
    best_pts = inits[0].copy()
    best_energy, _, _ = _energy_and_tangent_grad(best_pts)
    
    # Optimize each initialization using multiple methods
    n_restarts = len(inits)
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15:  # Reserve 15s for final polish
            break
        
        t_start = time.time()
        t_deadline = t_start + min(10.0, deadline - t_start - 10)
        
        try:
            # Method 1: L-BFGS-B
            optimized_pts = _optimize_lbfgs(init_pts, t_deadline, is_polish=False)
            optimized_pts = _project_to_sphere(optimized_pts)
            e_opt, _, _ = _energy_and_tangent_grad(optimized_pts)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = optimized_pts.copy()
                
        except Exception:
            pass
        
        # If time remains, try Riemannian L-BFGS from best found
        if time.time() < deadline - 10 and i % 2 == 0:
            try:
                t_deadline2 = min(deadline - 5, time.time() + 8.0)
                riem_pts = _riemannian_lbfgs(best_pts, t_deadline2, maxiter=1000)
                riem_pts = _project_to_sphere(riem_pts)
                e_riem, _, _ = _energy_and_tangent_grad(riem_pts)
                
                if e_riem < best_energy:
                    best_energy = e_riem
                    best_pts = riem_pts.copy()
                    
            except Exception:
                pass
    
    # Final high-precision polish on the best found
    if time.time() < deadline - 5:
        try:
            t_deadline = deadline - 2
            final_pts = _optimize_lbfgs(best_pts, t_deadline, is_polish=True, maxiter=10000)
            final_pts = _project_to_sphere(final_pts)
            e_final, _, _ = _energy_and_tangent_grad(final_pts)
            
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
                
        except Exception:
            pass
    
    # Try Riemannian CG as final refinement
    if time.time() < deadline - 3:
        try:
            t_deadline = deadline - 1
            cg_pts = _conjugate_gradient(best_pts, t_deadline, max_iter=500)
            cg_pts = _project_to_sphere(cg_pts)
            e_cg, _, _ = _energy_and_tangent_grad(cg_pts)
            
            if e_cg < best_energy:
                best_energy = e_cg
                best_pts = cg_pts.copy()
                
        except Exception:
            pass
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
