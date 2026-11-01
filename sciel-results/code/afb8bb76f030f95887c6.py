import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # Safety margin before 150s kill (leaves 35s buffer for overhead/return)

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    
    Gradient with respect to x_k:
    grad_k = -sum_{j!=k} (x_k - x_j) / ||x_k - x_j||^2
    """
    # points: (N, 3)
    # diff: (N, N, 3) where diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    
    # dist_sq: (N, N)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # inv_dist_sq: (N, N)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Gradient: grad[i] = -sum_j inv_dist_sq[i,j] * diff[i,j]
    # diff[i,j] = x_i - x_j
    # grad[i] = -sum_j (x_i - x_j) / ||x_i - x_j||^2
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy calculation
    # E = -0.5 * sum_{i!=j} log(dist_sq[i,j])  (since log(1/d) = -0.5 log(d^2) and we sum i<j, which is half of i!=j)
    # Actually, sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij) = -0.5 sum_{i<j} log(d_ij^2)
    # sum_{i!=j} log(d_ij^2) = 2 * sum_{i<j} log(d_ij^2)
    # So E = -0.25 * sum_{i!=j} log(d_ij^2) ? No.
    # E = sum_{i<j} -log(d_ij)
    # sum_{i<j} -log(d_ij) = -0.5 sum_{i<j} log(d_ij^2)
    # Let S = sum_{i!=j} log(d_ij^2). Then sum_{i<j} log(d_ij^2) = S/2.
    # So E = -0.5 * (S/2) = -0.25 * S.
    
    # However, it's safer to just sum the upper triangle to avoid factor errors.
    iu = np.triu_indices(N, k=1)
    d_sq_upper = dist_sq[iu]
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -np.sum(np.log(np.sqrt(d_sq_upper)))
    
    return energy, grad

def _tangent_grad(points, grad):
    """
    Project gradient onto the tangent space of the sphere at each point.
    T_p S^2 = { v : v . p = 0 }
    g_tan = g - (g . p) p
    """
    # grad: (N, 3), points: (N, 3)
    dot = np.sum(grad * points, axis=1, keepdims=True)
    return grad - dot * points

def _retract(points, step):
    """
    Retraction from tangent space to sphere.
    p_new = normalize(p + step)
    """
    p_new = points + step
    norms = np.linalg.norm(p_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return p_new / norms

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize_sphere(points, deadline):
    """
    Perform Riemannian optimization using L-BFGS-B on the tangent space.
    This is a simplified Riemannian L-BFGS where we linearize the manifold locally.
    """
    p = points.copy()
    e, g = _energy_and_grad(p)
    g_tan = _tangent_grad(p, g)
    
    # We will use a custom loop with L-BFGS-B on the flat vector of tangent vectors
    # This is a heuristic approach. A true Riemannian L-BFGS would be better, but this often works well.
    
    def _objective_and_grad_flat(x_flat):
        # x_flat is (3*N,) representing tangent vectors v_i
        v = x_flat.reshape(N, 3)
        # Move to sphere
        p_new = _retract(p, v)
        e_new, g_new = _energy_and_grad(p_new)
        g_tan_new = _tangent_grad(p_new, g_new)
        # The gradient of the energy w.r.t v is approximately g_tan_new
        # (Strictly, we need to account for the retraction, but for small steps it's close)
        return e_new, g_tan_new.flatten()
        
    x0 = np.zeros(3*N)
    
    try:
        # Run L-BFGS-B
        res = minimize(_objective_and_grad_flat, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        
        if res.success or res.status == 0:
            v_opt = res.x.reshape(N, 3)
            p_new = _retract(p, v_opt)
            e_new, _ = _energy_and_grad(p_new)
            if e_new < e:
                p = p_new
                e = e_new
                g = _energy_and_grad(p)[1]
                g_tan = _tangent_grad(p, g)
    except Exception:
        pass
    
    return p, e

def _polish_high_precision(points, deadline):
    """
    High-precision polish using Riemannian Steepest Descent with line search.
    """
    p = points.copy()
    e, g = _energy_and_grad(p)
    g_tan = _tangent_grad(p, g)
    
    best_p = p.copy()
    best_e = e
    
    step_size = 0.01
    max_iter = 10000
    tol = 1e-15
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        g_norm = np.linalg.norm(g_tan)
        if g_norm < tol:
            break
            
        # Line search
        step = step_size
        accepted = False
        c1 = 1e-4
        
        for _ in range(30):
            p_new = _retract(p, -step * g_tan)
            e_new, g_new = _energy_and_grad(p_new)
            
            if e_new < e - c1 * step * np.sum(g_tan**2):
                p = p_new
                e = e_new
                if e < best_e:
                    best_e = e
                    best_p = p.copy()
                g_tan = _tangent_grad(p, g_new)
                accepted = True
                break
            step *= 0.5
            
        if not accepted:
            step_size *= 0.5
            if step_size < 1e-10:
                break
        else:
            step_size = min(step_size * 1.05, 0.05)
            
    return best_p, best_e

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # 1. Initialization
    points = _fibonacci_spiral()
    best_points = points.copy()
    best_energy, _ = _energy_and_grad(best_points)
    
    # 2. Main Optimization Phase
    # Run optimization from Fibonacci start
    if time.time() < deadline - 20:
        p1, e1 = _optimize_sphere(points, deadline - 20)
        if e1 < best_energy:
            best_energy = e1
            best_points = p1
            
    # 3. Try a second initialization with random perturbation if time allows
    if time.time() < deadline - 20:
        rng = np.random.RandomState(42)
        points2 = _fibonacci_spiral()
        # Add small random noise to escape local minima
        noise = rng.randn(N, 3) * 0.1
        points2 = _retract(points2, noise)
        
        p2, e2 = _optimize_sphere(points2, deadline - 20)
        if e2 < best_energy:
            best_energy = e2
            best_points = p2

    # 4. High-Precision Polish
    if time.time() < deadline - 5:
        polished, e_polished = _polish_high_precision(best_points, deadline - 5)
        if e_polished < best_energy:
            best_energy = e_polished
            best_points = polished

    # 5. Final Validation
    # Ensure points are on the sphere and valid
    best_points = _retract(best_points, np.zeros((N, 3)))
    
    # Check for very close points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb slightly
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] = _retract(best_points[i], rng.randn(3) * 1e-5)
                
    return best_points
