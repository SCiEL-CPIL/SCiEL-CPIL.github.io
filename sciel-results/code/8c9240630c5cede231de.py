import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.25 * Sum_{i!=j} log(dist_sq_ij)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _project_to_sphere(pts):
    """
    Project points to the unit sphere.
    """
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, vecs):
    """
    Project vectors to the tangent space of the sphere at pts.
    pts: (N, 3)
    vecs: (N, 3)
    Returns: (N, 3) projected vectors
    """
    # Projection matrix P = I - x x^T
    # proj_vec = vec - (vec . x) x
    dot_products = np.sum(vecs * pts, axis=1, keepdims=True)
    return vecs - dot_products * pts

def _exp_map(pts, vecs):
    """
    Exponential map on the sphere.
    pts: (N, 3) points on sphere
    vecs: (N, 3) tangent vectors
    Returns: new points on sphere
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    
    theta = norms_safe
    # sin(theta)/theta ~ 1 for small theta
    sin_theta_over_theta = np.ones_like(theta)
    mask = theta > 1e-8
    sin_theta_over_theta[mask] = np.sin(theta[mask]) / theta[mask]
    
    # cos(theta)
    cos_theta = np.cos(theta)
    
    # x_new = cos(theta) * x + sin(theta)/theta * v
    new_pts = cos_theta * pts + sin_theta_over_theta * vecs
    return _project_to_sphere(new_pts)

def _riemannian_gradient(pts):
    """
    Compute the Riemannian gradient of the logarithmic energy on the sphere.
    pts: (N, 3)
    Returns: (N, 3) Riemannian gradient (tangent vectors)
    """
    # Euclidean gradient of E = -0.25 sum_{i!=j} log(d_ij^2)
    # dE/dx_i = -0.5 sum_{j!=i} (x_i - x_j) / d_ij^2
    
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Safety check
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return np.zeros_like(pts)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    # Riemannian gradient is the Euclidean gradient projected to the tangent space
    grad_riemann = _tangent_project(pts, grad_eucl)
    return grad_riemann

def _riemannian_adam(pts, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=2000):
    """
    Riemannian Adam optimization on the sphere.
    """
    pts = _project_to_sphere(pts)
    
    # Initialize moments
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    t = 0
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    for k in range(max_iter):
        t += 1
        
        # Compute gradient
        grad = _riemannian_gradient(pts)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        m_hat = m / (1 - beta1 ** t)
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad * grad)
        v_hat = v / (1 - beta2 ** t)
        
        # Compute update direction
        # Standard Adam: -lr * m_hat / (sqrt(v_hat) + eps)
        # We must ensure the update is in the tangent space
        update = -learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project update to tangent space (it should be, but numerical errors can creep in)
        update = _tangent_project(pts, update)
        
        # Line search (Backtracking) to ensure sufficient decrease
        # Armijo condition: f(x_new) <= f(x) + c * <grad, update>
        c1 = 1e-4
        rho = 0.5
        alpha = 1.0
        
        f_cur = _log_energy(pts)
        # Directional derivative: <grad, update>
        # Since update is tangent and grad is tangent, Euclidean dot is Riemannian inner product
        dir_deriv = np.sum(grad * update)
        
        # If dir_deriv is positive, we are moving uphill. This shouldn't happen with negative gradient,
        # but due to non-convexity and discrete steps, we might need to adjust.
        # However, Adam's update direction is approximately -grad, so dir_deriv should be negative.
        
        for _ in range(10):
            new_pts = _exp_map(pts, alpha * update)
            f_new = _log_energy(new_pts)
            
            if np.isfinite(f_new) and f_new <= f_cur + c1 * alpha * dir_deriv:
                break
            alpha *= rho
        else:
            # If line search fails, break
            break
            
        # Update points
        pts = new_pts
        f_cur = f_new
        
        if f_cur < best_energy:
            best_energy = f_cur
            best_pts = pts.copy()
            
        # Convergence check
        if np.linalg.norm(grad) < 1e-10:
            break
            
    return best_pts

def _md_init(n_points, steps=500, dt=0.01, temp=1.0):
    """
    Initialize points using a short Molecular Dynamics simulation.
    This helps find a configuration close to a local minimum by letting points repel each other.
    """
    # Start with random points
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(n_points, 3))
    pts = _project_to_sphere(pts)
    
    # Initialize velocities
    vels = rng.normal(size=(n_points, 3))
    vels = _tangent_project(pts, vels)
    
    for step in range(steps):
        # Compute forces (negative gradient)
        grad = _riemannian_gradient(pts)
        forces = -grad
        
        # Update velocities (Langevin dynamics for cooling)
        vels = 0.9 * vels + dt * forces + np.sqrt(2 * dt * temp) * rng.normal(size=(n_points, 3))
        vels = _tangent_project(pts, vels)
        
        # Update positions
        pts = _exp_map(pts, dt * vels)
        
        # Cool down temperature
        temp *= 0.99
        
    return pts

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _subdivided_icosahedron(n_points):
    """
    Generate points on the sphere using a subdivided icosahedron.
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 10], [0, 10, 7], [0, 7, 3],
        [0, 3, 4], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 3, 9], [3, 4, 8], [4, 2, 6], [2, 6, 8], [6, 8, 9],
        [2, 5, 9], [2, 9, 8], [11, 4, 2], [10, 2, 6], [7, 6, 9]
    ]
    
    def _icosa_subdivide(vertices, faces, levels):
        verts_list = [v.copy() for v in vertices]
        fs = [f.copy() for f in faces]
        
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (verts_list[v1] + verts_list[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts_list.append(mid)
                mid_cache[idx] = len(verts_list) - 1
                return mid_cache[idx]
                
            for f in fs:
                v1, v2, v3 = f
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                new_faces.append([v1, m12, m31])
                new_faces.append([v2, m23, m12])
                new_faces.append([v3, m31, m23])
                new_faces.append([m12, m23, m31])
                
            fs = new_faces
            
        return np.array(verts_list), fs

    # Level 2 gives 162 points. We want 150.
    verts, _ = _icosa_subdivide(verts, faces, 2)
    
    if len(verts) > n_points:
        # Remove points that are closest to their nearest neighbor
        min_dists = np.zeros(len(verts))
        for i in range(len(verts)):
            dists = np.linalg.norm(verts - verts[i], axis=1)
            np.fill_diagonal(dists, np.inf)
            min_dists[i] = np.min(dists)
        
        order = np.argsort(min_dists)
        indices_to_keep = order[len(verts)-n_points:]
        verts = verts[indices_to_keep]
        
    return verts

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Geodesic Dome (Subdivided Icosahedron)
    try:
        geo_pts = _subdivided_icosahedron(N)
        inits.append(geo_pts)
    except Exception:
        pass
        
    # 2. Fibonacci Sphere
    inits.append(_fibonacci_sphere(N))
    
    # 3. MD Initialized Points (Multiple seeds)
    for seed in [42, 123, 999]:
        try:
            # Modify random seed for MD
            np.random.seed(seed)
            md_pts = _md_init(N, steps=200, dt=0.005, temp=0.5)
            inits.append(md_pts)
        except Exception:
            pass
            
    # Optimization Loop using Riemannian Adam
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
            
        try:
            # Run Riemannian Adam
            opt_pts = _riemannian_adam(init_pts, learning_rate=0.005, max_iter=1000)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
            
    # If we have time, do a final polish on the best result using L-BFGS-B
    # L-BFGS is often better for the final tight convergence.
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            def _energy_and_grad(x_flat):
                pts = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                pts = pts / norms
                
                e = _log_energy(pts)
                if not np.isfinite(e):
                    return e, np.zeros_like(x_flat)
                    
                grad = _riemannian_gradient(pts)
                # For L-BFGS in Euclidean space, we need the Euclidean gradient
                # But since we are projecting, the effective gradient in the unconstrained space
                # is the Riemannian gradient projected back? 
                # Actually, standard L-BFGS on the sphere with projection works well if we use the Euclidean gradient.
                # Let's compute the Euclidean gradient again for L-BFGS.
                dot = pts @ pts.T
                dist_sq = 2.0 - 2.0 * dot
                np.fill_diagonal(dist_sq, 1.0)
                
                iu = np.triu_indices(N, k=1)
                d_sq = dist_sq[iu]
                if np.any(d_sq < 1e-12):
                    return e, np.zeros_like(x_flat)

                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                weights = -0.5 / dist_sq
                np.fill_diagonal(weights, 0.0)
                
                grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
                
                return e, grad_eucl.flatten()

            x0 = best_pts.flatten()
            res = minimize(
                _energy_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            opt_pts = res.x.reshape(N, 3)
            
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts.copy()
                
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print("Energy:", _log_energy(pts))
    print("Shape:", pts.shape)
