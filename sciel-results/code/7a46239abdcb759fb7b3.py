"""
Strategy: Deterministic Riemannian Newton-Krylov Method with Multi-Start.
1. Use Riemannian geometry (exponential maps) on the sphere manifold.
2. Implement Riemannian Newton's method using a Krylov subspace method (CG) 
   to solve the linear system for the search direction, avoiding explicit 
   Hessian storage.
3. Use multiple diverse initializations (Halton sequences) to explore the 
   landscape in parallel (sequentially in code, but distinct trajectories).
4. This leverages second-order convergence for high precision, unlike the 
   first-order SA/GD in the previous strategy.
"""

N = 150
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Using 0.5 * sum_{i!=j} -log(d_ij)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    # Note: The Euclidean gradient of the energy E = - sum_{i<j} log(||x_i-x_j||)
    # dE/dx_i = - sum_{j!=i} (1/||x_i-x_j||) * (x_i-x_j)/||x_i-x_j|| = - sum_{j!=i} (x_i-x_j)/||x_i-x_j||^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_project(grad, x):
    """
    Project Euclidean gradient to the tangent space of the sphere at x.
    proj_x(g) = g - (g . x) x
    """
    dot = np.sum(grad * x, axis=1, keepdims=True)
    return grad - dot * x

def _exponential_map(x, v):
    """
    Exponential map on the unit sphere.
    x: point on sphere (N, 3)
    v: tangent vector (N, 3)
    Returns: new point on sphere.
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    # Handle zero vectors
    mask = norm_v > 1e-10
    new_x = np.zeros_like(x)
    
    cos_theta = np.cos(norm_v)
    sin_theta_over_norm = np.zeros_like(norm_v)
    sin_theta_over_norm[mask] = np.sin(norm_v[mask]) / norm_v[mask]
    
    new_x = cos_theta * x + sin_theta_over_norm * v
    return new_x

def _hvp(x, v):
    """
    Computes the Riemannian Hessian-vector product.
    For a function f on the sphere, the Riemannian Hessian is related to the Euclidean Hessian 
    projected to the tangent space plus curvature terms.
    
    However, a simpler and robust approach for Newton-Krylov on manifolds is to use 
    the "Riemannian Hessian" defined via the covariant derivative.
    
    For the log energy E(x) = - sum_{i<j} log(||x_i - x_j||):
    The Riemannian Hessian H_r(v) at x applied to v is:
    H_r(v) = P_x ( H_eucl(v) + sum_{i} ( <grad_i, v_i> x_i - ... ) )
    
    Actually, let's use a numerical or semi-analytical approximation if the full analytic 
    Riemannian Hessian is complex. But for Newton to work well, we need a good approximation.
    
    Alternative: Use the Euclidean Hessian-vector product and project?
    The Euclidean Hessian H_eucl of E(x) is:
    H_eucl[i, j] (block) = delta_ij * sum_{k!=i} (I/||x_i-x_k||^2 - (x_i-x_k)(x_i-x_k)^T / ||x_i-x_k||^4)
                          - (1/||x_i-x_j||^2) I + (x_i-x_j)(x_i-x_j)^T / ||x_i-x_j||^4  for i!=j
    
    Wait, the Hessian of -log(||x||) is -1/||x||^2 I + 2 x x^T / ||x||^4.
    So for pair (i,j), contribution to Hessian block (i,j) is:
    - ( -1/||d||^2 I + 2 d d^T / ||d||^4 ) = 1/||d||^2 I - 2 d d^T / ||d||^4
    
    And for diagonal block (i,i), sum over all j!=i of:
    -1/||d||^2 I + 2 d d^T / ||d||^4  (Note: gradient of -log(d) is -d/d^2. Hessian of -log(d) is -1/d^2 I + 2 dd^T/d^4)
    
    Let's compute the Euclidean Hessian-vector product H_eucl * v.
    """
    # v is (N, 3)
    # diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    
    # We need to compute H_eucl * v
    # H_eucl is a block matrix.
    # Block (i, j) for i != j: 1/||d_ij||^2 I - 2 d_ij d_ij^T / ||d_ij||^4
    # Block (i, i): sum_{k!=i} ( -1/||d_ik||^2 I + 2 d_ik d_ik^T / ||d_ik||^4 )
    
    # Let's compute the action on v.
    # (H * v)_i = sum_j H_ij v_j
    
    # Initialize result
    Hv = np.zeros_like(x)
    
    # Off-diagonal terms
    # H_ij v_j = (1/||d_ij||^2) v_j - 2 (d_ij . v_j) d_ij / ||d_ij||^4
    # Sum over j for fixed i
    
    # Vectorized computation
    # dist_sq is (N, N)
    # diff is (N, N, 3)
    
    # 1/||d||^2 * v
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    term1 = inv_dist_sq[:, :, np.newaxis] * v[np.newaxis, :, :] # (N, N, 3) -> sum over j gives (N, 3)
    # term1[i] = sum_j (1/||d_ij||^2) v_j
    
    # 2 (d_ij . v_j) d_ij / ||d_ij||^4
    # dot_dv = sum_k diff[i,j,k] * v[j,k]
    dot_dv = np.sum(diff * v[np.newaxis, :, :], axis=2) # (N, N)
    
    inv_dist_sq_2 = np.zeros_like(dist_sq)
    inv_dist_sq_2[mask] = 2.0 / (dist_sq[mask] ** 2)
    
    term2 = dot_dv[:, :, np.newaxis] * diff * inv_dist_sq_2[:, :, np.newaxis] # (N, N, 3)
    # term2[i] = sum_j 2 (d_ij . v_j) d_ij / ||d_ij||^4
    
    off_diag_contrib = np.sum(term1 - term2, axis=1)
    
    # Diagonal terms
    # H_ii v_i = sum_{k!=i} ( -1/||d_ik||^2 v_i + 2 (d_ik . v_i) d_ik / ||d_ik||^4 )
    # Note: d_ik = x_i - x_k
    # term1_diag[i] = - (sum_{k!=i} 1/||d_ik||^2) * v_i
    # term2_diag[i] = sum_{k!=i} 2 (d_ik . v_i) d_ik / ||d_ik||^4
    
    # sum_inv_dist_sq_i = sum_{k!=i} 1/||d_ik||^2
    sum_inv_dist_sq = np.sum(inv_dist_sq, axis=1) # (N,)
    
    diag_term1 = -sum_inv_dist_sq[:, np.newaxis] * v
    
    # dot_di_vi = sum_k diff[i,k] . v[i] = v[i] . sum_k diff[i,k] ? No, v[i] is constant for the sum over k.
    # dot_di_vi = v[i] . (sum_{k!=i} d_ik)
    # Let S_i = sum_{k!=i} d_ik
    S = np.sum(diff, axis=1) # (N, 3)
    dot_di_vi = np.sum(v * S, axis=1, keepdims=True) # (N, 1)
    
    # term2_diag[i] = sum_{k!=i} 2 (d_ik . v_i) d_ik / ||d_ik||^4
    # This is not simply (v_i . S_i) * something because of the 1/||d||^4 weighting.
    # We need to sum k for each i.
    # Weighted sum: sum_k (2 * (diff[i,k] . v[i]) / ||d_ik||^4) * diff[i,k]
    
    # dot_diff_v = sum_k diff[i,k] * v[i,k] -> same for all k? No, v[i] is fixed for row i.
    # dot_diff_v[i,k] = sum_m diff[i,k,m] * v[i,m]
    dot_diff_v = np.einsum('ijk,ik->ij', diff, v) # (N, N)
    
    # coeff[i,k] = 2 * dot_diff_v[i,k] / dist_sq[i,k]^2
    coeff = np.zeros_like(dist_sq)
    coeff[mask] = 2.0 * dot_diff_v[mask] / (dist_sq[mask] ** 2)
    
    diag_term2 = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    Hv = off_diag_contrib + diag_term1 + diag_term2
    
    return Hv

def _riemannian_hvp(x, v):
    """
    Approximate the Riemannian Hessian-vector product.
    A common approximation is P_x ( H_eucl(v) + <grad, v> x )
    where P_x is projection to tangent space.
    """
    # H_eucl * v
    H_eucl_v = _hvp(x, v)
    
    # <grad, v> x
    _, grad = _log_energy_and_grad_sphere(x)
    grad_v_dot = np.sum(grad * v, axis=1, keepdims=True)
    
    # Sum term
    Hv_r = H_eucl_v + grad_v_dot * x
    
    # Project to tangent space
    Hv_r = _riemannian_project(Hv_r, x)
    
    return Hv_r

def _riemannian_newton_krylov(x, max_iter=50, tol=1e-10):
    """
    Riemannian Newton's method using Conjugate Gradient to solve for the step.
    Solves H_r p = -grad_r in the tangent space.
    """
    x_curr = x.copy()
    energy, grad = _log_energy_and_grad_sphere(x_curr)
    rgrad = _riemannian_project(grad, x_curr)
    
    # Check if gradient is small
    if np.linalg.norm(rgrad) < tol:
        return x_curr
    
    for i in range(max_iter):
        # Solve H_r p = -rgrad using CG
        # We need to solve H p = b where b = -rgrad
        # H is the Riemannian Hessian operator.
        
        # Initialize CG
        p = -rgrad.copy()
        r = -rgrad.copy()
        rs_old = np.sum(r * r)
        
        if rs_old < tol:
            break
            
        for k in range(20): # CG iterations
            Hp = _riemannian_hvp(x_curr, p)
            # Ensure Hp is in tangent space (it should be by definition, but numerical errors)
            Hp = _riemannian_project(Hp, x_curr)
            
            alpha = rs_old / np.sum(p * Hp)
            
            if not np.isfinite(alpha) or alpha <= 0:
                # Fallback to steepest descent step if Hessian is not positive definite or numerical issues
                p = -rgrad
                alpha = 0.1
                break
            
            x_new = _exponential_map(x_curr, alpha * p)
            
            # Check if step is valid (energy decreases)
            e_new, grad_new = _log_energy_and_grad_sphere(x_new)
            
            if e_new < energy:
                # Accept step
                x_curr = x_new
                energy = e_new
                rgrad = _riemannian_project(grad_new, x_curr)
                break
            else:
                # Damp step
                alpha *= 0.5
                x_new = _exponential_map(x_curr, alpha * p)
                e_new, grad_new = _log_energy_and_grad_sphere(x_new)
                if e_new < energy:
                    x_curr = x_new
                    energy = e_new
                    rgrad = _riemannian_project(grad_new, x_curr)
                    break
                else:
                    # If still not decreasing, switch to steepest descent for this iteration
                    p = -rgrad
                    alpha = 0.01
                    # Simple line search for steepest descent
                    for ls in range(10):
                        x_try = _exponential_map(x_curr, alpha * p)
                        e_try, _ = _log_energy_and_grad_sphere(x_try)
                        if e_try < energy:
                            x_curr = x_try
                            energy = e_try
                            _, grad_try = _log_energy_and_grad_sphere(x_curr)
                            rgrad = _riemannian_project(grad_try, x_curr)
                            break
                        alpha *= 0.5
                    break
        
        if np.linalg.norm(rgrad) < tol:
            break
        
    return x_curr

def _halton_sphere(n, base1=2, base2=3):
    """Generate Halton sequence points on the sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(n):
        # Shift index to avoid 0,0
        idx = i + 1
        u1 = halton(idx, base1)
        u2 = halton(idx, base2)
        
        # Map to sphere using inverse CDF of uniform distribution on sphere
        # z = 2*u1 - 1
        # theta = 2*pi*u2
        z = 2.0 * u1 - 1.0
        theta = 2.0 * math.pi * u2
        r = math.sqrt(1.0 - z*z)
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append([x, y, z])
        
    return np.array(points)

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N, 2, 3),
        _halton_sphere(N, 3, 5),
        _halton_sphere(N, 5, 7)
    ]
    
    best_energy = float('inf')
    best_x = None
    
    # Run Newton-Krylov on each initialization
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x_curr = x_init.copy()
        
        # Run Riemannian Newton
        x_opt = _riemannian_newton_krylov(x_curr)
        
        e_opt, _ = _log_energy_and_grad_sphere(x_opt)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
            
    # If we have time, do a final polish on the best result
    if best_x is not None and time.time() < deadline - 2:
        x_final = _riemannian_newton_krylov(best_x)
        e_final, _ = _log_energy_and_grad_sphere(x_final)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final
            
    return best_x

# EVOLVE-BLOCK-END
