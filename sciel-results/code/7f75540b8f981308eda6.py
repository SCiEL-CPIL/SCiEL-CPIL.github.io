"""
Minimize logarithmic energy of 100 points on the unit sphere.
Uses L-BFGS-B with multiple diverse initializations and Newton-Raphson polishing.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|p_i - p_j|)
    Gradient: sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2, projected to tangent space.
    """
    n = len(pts)
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Compute squared distances
    d2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j (distance 0)
    # We will mask these out in summation
    np.fill_diagonal(d2, 1.0)
    
    # Distances
    d = np.sqrt(d2)
    
    # Energy calculation
    # We only need i < j
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask_upper]
    
    # Check for extremely small distances (points coinciding)
    if np.any(d_upper < 1e-9):
        # This is bad, return high energy to push away
        return 1e18, np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    # Gradient calculation
    # grad_i = sum_{j != i} (p_i - p_j) / |p_i - p_j|^2
    # Note: diff[i,j] = p_i - p_j
    # So term is diff[i,j] / d2[i,j]
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j (p_i - p_j) / d2_ij
    # Vectorized: sum over j (axis=1)
    grad = np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient onto tangent space
    # p_i is unit vector, so normal is p_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial * pts
    
    return energy, grad

def _hessian_tangent(pts):
    """
    Compute the Hessian of the log-energy on the tangent space.
    For the log kernel V(r) = -log(r), the Hessian in 3D is:
    H_ij = sum_{k != i,j} [ (delta_ij / r_k^2) * I - (r_k * r_k^T) / r_k^4 ]
    where r_k = p_i - p_k for the i-th row block.
    
    Actually, let's be more careful. The energy is E = sum_{i<j} -log(|p_i - p_j|).
    The Hessian with respect to p_i and p_j (i != j) is:
    d^2 E / dp_i dp_j = -[ I/|p_i-p_j|^2 - (p_i-p_j)(p_i-p_j)^T / |p_i-p_j|^4 ]
    
    For i = j:
    d^2 E / dp_i^2 = sum_{k != i} [ I/|p_i-p_k|^2 - (p_i-p_k)(p_i-p_k)^T / |p_i-p_k|^4 ]
    
    We need the projected Hessian on the tangent space.
    """
    n = len(pts)
    # This is a 3n x 3n matrix, but we'll work with the full Hessian and project
    # For efficiency with n=100, we compute the 300x300 Hessian
    
    H = np.zeros((n*3, n*3))
    
    for i in range(n):
        for j in range(i+1, n):
            rij = pts[i] - pts[j]
            d2 = np.dot(rij, rij)
            if d2 < 1e-18:
                continue
            d4 = d2 * d2
            # The Hessian block for (i,j) and (j,i)
            # d^2(-log|rij|)/dri^2 = I/d2 - rij*rij^T/d4
            # d^2(-log|rij|)/dri drj = -I/d2 + rij*rij^T/d4
            block_ii = np.eye(3) / d2 - np.outer(rij, rij) / d4
            block_ij = -np.eye(3) / d2 + np.outer(rij, rij) / d4
            
            # Place in H
            H[i*3:(i+1)*3, i*3:(i+1)*3] += block_ii
            H[j*3:(j+1)*3, j*3:(j+1)*3] += block_ii
            H[i*3:(i+1)*3, j*3:(j+1)*3] += block_ij
            H[j*3:(j+1)*3, i*3:(i+1)*3] += block_ij
    
    # Project to tangent space
    # For each point i, the tangent projection is P_i = I - p_i p_i^T
    # The projected Hessian H_t[i,j] = P_i @ H[i,j] @ P_j
    H_t = np.zeros_like(H)
    for i in range(n):
        Pi = np.eye(3) - np.outer(pts[i], pts[i])
        for j in range(n):
            Pj = np.eye(3) - np.outer(pts[j], pts[j])
            H_t[i*3:(i+1)*3, j*3:(j+1)*3] = Pi @ H[i*3:(i+1)*3, j*3:(j+1)*3] @ Pj
    
    return H_t

def _newton_polish(pts0, max_iter=20, damping=0.5):
    """
    Newton-Raphson polishing on the sphere using tangent space Hessian.
    """
    pts = pts0.copy()
    n = len(pts)
    
    for it in range(max_iter):
        e, g = _energy_and_grad(pts)
        if np.linalg.norm(g, ord=np.inf) < 1e-14:
            break
        
        H = _hessian_tangent(pts)
        
        # Solve H @ delta = -g in tangent space
        # Flatten g to 3n vector
        g_flat = g.flatten()
        
        # Add small regularization for numerical stability
        reg = 1e-10
        H_reg = H + reg * np.eye(n*3)
        
        try:
            delta_flat = np.linalg.solve(H_reg, -g_flat)
        except np.linalg.LinAlgError:
            delta_flat = np.linalg.lstsq(H_reg, -g_flat, rcond=None)[0]
        
        # Project delta to tangent space for each point
        delta = delta_flat.reshape(n, 3)
        for i in range(n):
            # Project to tangent space
            radial = np.dot(delta[i], pts[i])
            delta[i] -= radial * pts[i]
        
        # Damped step
        step = 1.0
        for _ in range(20):
            # Exponential map: move along geodesic
            new_pts = np.zeros_like(pts)
            for i in range(n):
                d = delta[i]
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-15:
                    new_pts[i] = pts[i]
                else:
                    # Exponential map on sphere
                    angle = step * d_norm
                    if abs(angle) < 1e-15:
                        new_pts[i] = pts[i]
                    else:
                        v = d / d_norm
                        new_pts[i] = np.cos(angle) * pts[i] + np.sin(angle) * v
                        # Normalize
                        new_pts[i] /= np.linalg.norm(new_pts[i])
            
            e_new, _ = _energy_and_grad(new_pts)
            if e_new < e:
                pts = new_pts
                break
            step *= 0.5
        
        if step < 1e-10:
            break
    
    return pts, _energy_and_grad(pts)[0]

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosahedron_sphere(n):
    """Generate points based on icosahedron subdivision."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide and project to sphere
    # Use Fibonacci for the rest to fill up to n
    if n <= 12:
        return verts[:n]
    
    # Start with icosahedron, then add Fibonacci points
    pts = [verts]
    remaining = n - 12
    fib = _fibonacci_sphere(remaining)
    all_pts = np.vstack([verts, fib])
    
    # Normalize
    all_pts /= np.linalg.norm(all_pts, axis=1, keepdims=True)
    return all_pts

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _optimize_lbfgs(pts0, deadline, maxiter=5000):
    """
    Optimize using L-BFGS-B.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 
                     'ftol': 1e-15, 'gtol': 1e-10, 'm': 20}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def _fine_polish(pts0, deadline, maxiter=10000):
    """
    High precision polishing using L-BFGS-B with very tight tolerances.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        _, g = _energy_and_grad(pts)
        return g.flatten()

    try:
        result = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                     'ftol': 1e-16, 'gtol': 1e-12, 'm': 10}
        )
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _energy_and_grad(best_pts)[0]
            return best_pts, best_energy
        else:
            return pts0, _energy_and_grad(pts0)[0]
    except Exception:
        return pts0, _energy_and_grad(pts0)[0]

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. Keep 10s buffer.
    deadline = start_time + 140.0

    best_pts = None
    best_energy = np.inf

    # 1. Diverse Initializations
    # Fibonacci sphere is a very strong heuristic for this problem.
    initializations = []
    
    # Primary: Fibonacci
    initializations.append(_fibonacci_sphere(N))
    
    # Secondary: Icosahedron-based
    initializations.append(_icosahedron_sphere(N))
    
    # Tertiary: Random starts with different seeds
    seeds = [42, 123, 777, 999, 2023]
    for s in seeds:
        initializations.append(_random_sphere(N, s))

    # 2. Optimization Loop
    for i, pts0 in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        # Standard L-BFGS optimization
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline, maxiter=3000)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
            
        # Early termination if we hit a very low energy (unlikely but possible)
        # The record is around -1083.37
        if best_energy < -1083.377:
            # We have beaten the published record (if true) or are very close.
            # Continue polishing below.
            pass

    if best_pts is None:
        # Fallback: return Fibonacci if optimization failed
        return _fibonacci_sphere(N)

    # 3. Polishing Phase
    # Use remaining time to refine the best candidate
    time_left = deadline - time.time()
    
    if time_left > 10.0:
        # High precision polish with L-BFGS-B
        pts_polished, e_polished = _fine_polish(best_pts, deadline, maxiter=5000)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished
    
    # 4. Newton-Raphson Polish
    # This should converge to machine precision
    if time.time() < deadline - 5.0:
        try:
            pts_newton, e_newton = _newton_polish(best_pts, max_iter=30)
            if e_newton < best_energy:
                best_energy = e_newton
                best_pts = pts_newton
        except Exception:
            pass

    # 5. Final Check and Return
    # Ensure points are strictly on the sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
