"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    d_ij = ||p_i - p_j||
    
    Gradient with respect to p_i:
    grad_i = sum_{j != i} (p_i - p_j) / d_ij^2
    (Note: The derivative of -log(d) is -1/d * d/dp_i (d) = -1/d * (p_i - p_j)/d = -(p_i - p_j)/d^2)
    Wait, let's re-verify.
    E = - sum_{i<j} ln(||p_i - p_j||)
    dE/dp_k = - sum_{j != k} (1/||p_k - p_j||) * (1/||p_k - p_j||) * (p_k - p_j)
            = - sum_{j != k} (p_k - p_j) / ||p_k - p_j||^2
    
    This is the gradient in Euclidean space.
    To project onto the tangent space of the sphere (since points must remain on the sphere),
    we project the gradient: grad_tangent = grad - (grad . p_k) * p_k
    Since p_k is unit length, p_k . p_k = 1.
    """
    if not np.all(np.isfinite(points)):
        return float('inf'), np.zeros_like(points)
    
    # Compute pairwise distances
    # dists[i, j] = ||p_i - p_j||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf'), np.zeros_like(points)
        
    # Energy
    # Only sum upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient
    # grad[i] = - sum_{j != i} (p_i - p_j) / d_ij^2
    # We can compute this using matrix operations or loops.
    # Let's use a loop for clarity and safety, N=100 is small.
    grad = np.zeros_like(points)
    for i in range(N):
        # Vectorized over j for fixed i
        # d_ij is dists[i, :]
        # p_i - p_j is points[i] - points
        # term = (points[i] - points) / (dists[i, :]**2)[:, np.newaxis]
        # But we need to exclude j=i. dists[i,i] is 1.0, so term[i] is 0.
        # However, the formula is valid for j != i.
        # Let's compute the sum for all j, then subtract the i-th term (which is 0 anyway if we mask).
        
        # Actually, simpler:
        # diff = points[i] - points  # shape (N, 3)
        # dist_sq = dists[i, :]**2   # shape (N,)
        # term = diff / dist_sq[:, np.newaxis]
        # term[i] = 0
        # grad[i] = -np.sum(term, axis=0)
        
        diff_i = points[i] - points
        dist_sq = dists[i, :]**2
        # Avoid division by zero just in case, though diagonal is 1.
        dist_sq[i] = 1.0 
        term = diff_i / dist_sq[:, np.newaxis]
        term[i] = 0.0
        grad[i] = -np.sum(term, axis=0)
        
    # Project gradient onto tangent space
    # grad_tangent[i] = grad[i] - (grad[i] . points[i]) * points[i]
    for i in range(N):
        dot = np.dot(grad[i], points[i])
        grad[i] -= dot * points[i]
        
    return energy, grad

def _sphere_energy_and_grad(points):
    """
    Calculate the logarithmic energy and its Riemannian gradient on the sphere.
    Returns (energy, tangent_gradient) where tangent_gradient is in the tangent space at each point.
    """
    if not np.all(np.isfinite(points)):
        return float('inf'), np.zeros_like(points)
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Check for invalid close points
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return float('inf'), np.zeros_like(points)
        
    # Energy
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    # Riemannian gradient (tangent space)
    # For each point i, the Riemannian gradient is the projection of the Euclidean gradient
    # onto the tangent space at p_i.
    # Euclidean gradient of E with respect to p_i is:
    #   grad_E[i] = - sum_{j != i} (p_i - p_j) / d_ij^2
    # Riemannian gradient = grad_E[i] - (grad_E[i] . p_i) * p_i
    
    grad_riemannian = np.zeros_like(points)
    for i in range(N):
        diff_i = points[i] - points
        dist_sq = dists[i, :]**2
        dist_sq[i] = 1.0 
        term = diff_i / dist_sq[:, np.newaxis]
        term[i] = 0.0
        grad_euclidean_i = -np.sum(term, axis=0)
        
        # Project onto tangent space
        dot = np.dot(grad_euclidean_i, points[i])
        grad_riemannian[i] = grad_euclidean_i - dot * points[i]
        
    return energy, grad_riemannian

def _retract(points, tangent):
    """
    Retraction map from tangent space to the sphere.
    Given points on the sphere and a tangent vector at each point,
    compute new points on the sphere.
    Uses exponential map approximation: p_new = (p + v) / ||p + v||
    where v is the tangent vector.
    """
    new_points = points + tangent
    norms = np.linalg.norm(new_points, axis=1)
    # Handle case where norm is zero (shouldn't happen if tangent is small)
    mask = norms > 1e-12
    new_points[mask] = new_points[mask] / norms[mask, np.newaxis]
    # For zero norms, keep original point
    new_points[~mask] = points[~mask]
    return new_points

def _lbfgs_sphere(points, deadline, maxiter=300, gtol=1e-10):
    """
    L-BFGS optimization on the sphere using retraction.
    Works in the tangent space at each iteration.
    """
    current = points.copy()
    best_points = points.copy()
    best_energy = _sphere_energy_and_grad(points)[0]
    
    if not np.isfinite(best_energy):
        return best_points, best_energy
    
    # L-BFGS memory
    m = 10  # memory size
    s_list = []  # list of tangent vectors
    y_list = []  # list of gradient differences
    rho_list = []  # list of 1/(y^T s)
    
    # Initial gradient
    e0, g0 = _sphere_energy_and_grad(current)
    if not np.isfinite(e0):
        return best_points, best_energy
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        # Check convergence
        gnorm = np.linalg.norm(g0)
        if gnorm < gtol:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        # Flatten for L-BFGS
        g_flat = g0.flatten()
        
        # Two-loop recursion for direction
        q = g_flat.copy()
        alphas = np.zeros(len(s_list))
        
        for k in range(len(s_list) - 1, -1, -1):
            s_k = s_list[k].flatten()
            y_k = y_list[k].flatten()
            rho_k = rho_list[k]
            alpha_k = rho_k * np.dot(s_k, q)
            alphas[k] = alpha_k
            q -= alpha_k * y_k
        
        # Scale by inverse of last y^T s
        if len(s_list) > 0:
            s_last = s_list[-1].flatten()
            y_last = y_list[-1].flatten()
            gamma = np.dot(s_last, y_last)
            if gamma > 1e-12:
                r = gamma / np.dot(y_last, y_last) * q
            else:
                r = q
        else:
            r = q.copy()
        
        for k in range(len(s_list)):
            s_k = s_list[k].flatten()
            y_k = y_list[k].flatten()
            rho_k = rho_list[k]
            beta_k = rho_k * np.dot(y_k, r)
            r += (alphas[k] - beta_k) * s_k
        
        direction = -r.reshape(N, 3)
        
        # Line search
        alpha = 1.0
        c1 = 1e-4
        max_ls = 20
        found_alpha = False
        
        for ls_iter in range(max_ls):
            if time.time() > deadline:
                break
            
            # Compute new point
            new_points = _retract(current, alpha * direction)
            
            # Check validity
            if not np.all(np.isfinite(new_points)):
                alpha *= 0.5
                continue
            
            # Compute energy and gradient at new point
            e_new, g_new = _sphere_energy_and_grad(new_points)
            
            if not np.isfinite(e_new):
                alpha *= 0.5
                continue
            
            # Armijo condition
            if e_new <= e0 + c1 * alpha * np.sum(g0.flatten() * direction.flatten()):
                found_alpha = True
                break
            
            alpha *= 0.5
        
        if not found_alpha or time.time() > deadline:
            break
        
        # Update L-BFGS memory
        s = alpha * direction
        y = g_new - g0
        
        # Check curvature condition
        y_norm_sq = np.sum(y.flatten()**2)
        s_dot_y = np.dot(s.flatten(), y.flatten())
        
        if y_norm_sq > 1e-12 and s_dot_y > 1e-12 * y_norm_sq:
            rho = 1.0 / s_dot_y
            s_list.append(s)
            y_list.append(y)
            rho_list.append(rho)
            
            # Maintain memory size
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
        
        # Update current point
        current = new_points
        e0 = e_new
        g0 = g_new
        
        if e0 < best_energy:
            best_energy = e0
            best_points = current.copy()
    
    return best_points, best_energy

def _polish(points, deadline, maxiter=200, gtol=1e-12):
    """
    High-precision polish using gradient descent with adaptive step size.
    """
    current = points.copy()
    best_points = points.copy()
    best_energy = _sphere_energy_and_grad(points)[0]
    
    if not np.isfinite(best_energy):
        return best_points, best_energy
    
    e0, g0 = _sphere_energy_and_grad(current)
    if not np.isfinite(e0):
        return best_points, best_energy
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        gnorm = np.linalg.norm(g0)
        if gnorm < gtol:
            break
        
        # Adaptive step size
        alpha = min(0.1, 0.1 / (gnorm + 1e-8))
        
        # Try a few step sizes
        found = False
        for trial in range(10):
            new_points = _retract(current, alpha * g0)
            
            if not np.all(np.isfinite(new_points)):
                alpha *= 0.5
                continue
            
            e_new, g_new = _sphere_energy_and_grad(new_points)
            
            if not np.isfinite(e_new):
                alpha *= 0.5
                continue
            
            # Decrease condition
            if e_new < e0:
                found = True
                break
            
            alpha *= 0.5
        
        if not found or time.time() > deadline:
            break
        
        current = new_points
        e0 = e_new
        g0 = g_new
        
        if e0 < best_energy:
            best_energy = e0
            best_points = current.copy()
    
    return best_points, best_energy

def _ultra_polish(points, deadline, maxiter=500, gtol=1e-14):
    """
    Ultra-high-precision polish with very tight tolerance.
    """
    current = points.copy()
    best_points = points.copy()
    best_energy = _sphere_energy_and_grad(points)[0]
    
    if not np.isfinite(best_energy):
        return best_points, best_energy
    
    e0, g0 = _sphere_energy_and_grad(current)
    if not np.isfinite(e0):
        return best_points, best_energy
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        gnorm = np.linalg.norm(g0)
        if gnorm < gtol:
            break
        
        # Very small step sizes
        alpha = min(0.01, 0.01 / (gnorm + 1e-10))
        
        found = False
        for trial in range(15):
            new_points = _retract(current, alpha * g0)
            
            if not np.all(np.isfinite(new_points)):
                alpha *= 0.5
                continue
            
            e_new, g_new = _sphere_energy_and_grad(new_points)
            
            if not np.isfinite(e_new):
                alpha *= 0.5
                continue
            
            if e_new < e0:
                found = True
                break
            
            alpha *= 0.5
        
        if not found or time.time() > deadline:
            break
        
        current = new_points
        e0 = e_new
        g0 = g_new
        
        if e0 < best_energy:
            best_energy = e0
            best_points = current.copy()
    
    return best_points, best_energy

def _hessian_and_rhessian(points):
    """
    Compute the Riemannian Hessian of the energy function on the sphere.
    Returns the Hessian as a 300x300 matrix (flattened) and the energy/gradient.
    
    For E = -sum_{i<j} log(||p_i - p_j||), the Euclidean Hessian block (i,j) is:
      H_ij = delta_ij * sum_{k != i} [I/d_ik^2 - 3*(p_i-p_k)(p_i-p_k)^T/d_ik^4]
             - (I/d_ij^2 - 3*(p_i-p_j)(p_i-p_j)^T/d_ij^4)  for i != j
    
    The Riemannian Hessian projects this onto tangent spaces.
    """
    e, g = _sphere_energy_and_grad(points)
    if not np.isfinite(e):
        return None, None, None
    
    # Compute pairwise distances and squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0)
    
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        return None, None, None
    
    d_sq = dists ** 2
    np.fill_diagonal(d_sq, 1.0)
    
    # Build the Euclidean Hessian H (N x N x 3 x 3)
    # H[i,j] = sum_k contribution
    # For i == j: H_ii = sum_{k != i} [I/d_ik^2 - 3*(p_i-p_k)(p_i-p_k)^T/d_ik^4]
    # For i != j: H_ij = -[I/d_ij^2 - 3*(p_i-p_j)(p_i-p_j)^T/d_ij^4]
    
    # We'll build it block by block
    H = np.zeros((N, N, 3, 3))
    
    for i in range(N):
        for j in range(i, N):
            if i == j:
                # Diagonal block: sum over k != i
                block = np.zeros((3, 3))
                for k in range(N):
                    if k == i:
                        continue
                    d_ik = dists[i, k]
                    d_ik_sq = d_sq[i, k]
                    d_ik_4 = d_ik_sq * d_ik_sq
                    diff_ik = points[i] - points[k]
                    outer = np.outer(diff_ik, diff_ik)
                    block += np.eye(3) / d_ik_sq - 3.0 * outer / d_ik_4
                H[i, i] = block
                H[j, j] = block
            else:
                # Off-diagonal block
                d_ij = dists[i, j]
                d_ij_sq = d_sq[i, j]
                d_ij_4 = d_ij_sq * d_ij_sq
                diff_ij = points[i] - points[j]
                outer = np.outer(diff_ij, diff_ij)
                block = -(np.eye(3) / d_ij_sq - 3.0 * outer / d_ij_4)
                H[i, j] = block
                H[j, i] = block
    
    # Now compute the Riemannian Hessian
    # The Riemannian Hessian at point p_i in direction v_i is:
    #   Hess_R[i] = P_i * (H[i,i] * v_i + sum_j H[i,j] * v_j) - (p_i . H[i,i] * v_i) * p_i
    # where P_i = I - p_i p_i^T
    # 
    # For the Newton step, we need to solve:
    #   sum_j (P_i H_ij P_j) v_j = -g_i
    # 
    # This is a 300x300 system. Let's build the projected Hessian matrix.
    
    # Flatten to 300x300
    H_flat = np.zeros((3*N, 3*N))
    
    for i in range(N):
        P_i = np.eye(3) - np.outer(points[i], points[i])
        for j in range(N):
            P_j = np.eye(3) - np.outer(points[j], points[j])
            # Projected block: P_i @ H[i,j] @ P_j
            block = P_i @ H[i, j] @ P_j
            H_flat[i*3:(i+1)*3, j*3:(j+1)*3] = block
    
    return H_flat, e, g

def _newton_polish(points, deadline, maxiter=30):
    """
    Newton polish using the Riemannian Hessian for quadratic convergence.
    Uses Cholesky decomposition with fallback to regularized solve.
    """
    current = points.copy()
    best_points = points.copy()
    best_energy = _sphere_energy_and_grad(points)[0]
    
    if not np.isfinite(best_energy):
        return best_points, best_energy
    
    e0, g0 = _sphere_energy_and_grad(current)
    if not np.isfinite(e0):
        return best_points, best_energy
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        gnorm = np.linalg.norm(g0)
        if gnorm < 1e-13:
            break
        
        # Compute Riemannian Hessian
        H_flat, e_h, g_h = _hessian_and_rhessian(current)
        if H_flat is None:
            break
        
        # Solve H * v = -g for the Newton step
        # Add regularization for numerical stability
        reg = 1e-8
        H_reg = H_flat + reg * np.eye(3*N)
        
        try:
            # Try Cholesky first
            L = np.linalg.cholesky(H_reg)
            v = np.linalg.solve(L.T, np.linalg.solve(L, -g0.flatten()))
        except np.linalg.LinAlgError:
            try:
                v = np.linalg.solve(H_reg, -g0.flatten())
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                v = np.linalg.lstsq(H_reg, -g0.flatten(), rcond=None)[0]
        
        v = v.reshape(N, 3)
        
        # Line search with backtracking
        alpha = 1.0
        found = False
        for trial in range(25):
            if time.time() > deadline:
                break
            
            new_points = _retract(current, alpha * v)
            
            if not np.all(np.isfinite(new_points)):
                alpha *= 0.5
                continue
            
            e_new, g_new = _sphere_energy_and_grad(new_points)
            
            if not np.isfinite(e_new):
                alpha *= 0.5
                continue
            
            # Armijo condition
            if e_new <= e0 + 1e-4 * alpha * np.sum(g0.flatten() * v.flatten()):
                found = True
                break
            
            alpha *= 0.5
        
        if not found or time.time() > deadline:
            break
        
        current = new_points
        e0 = e_new
        g0 = g_new
        
        if e0 < best_energy:
            best_energy = e0
            best_points = current.copy()
    
    return best_points, best_energy

def _rcg_polish(points, deadline, maxiter=200):
    """
    Riemannian conjugate gradient polish.
    """
    current = points.copy()
    best_points = points.copy()
    best_energy = _sphere_energy_and_grad(points)[0]
    
    if not np.isfinite(best_energy):
        return best_points, best_energy
    
    e0, g0 = _sphere_energy_and_grad(current)
    if not np.isfinite(e0):
        return best_points, best_energy
    
    # Initial direction
    d = -g0
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        gnorm = np.linalg.norm(g0)
        if gnorm < 1e-12:
            break
        
        # Line search
        alpha = 1.0
        found = False
        for trial in range(20):
            new_points = _retract(current, alpha * d)
            
            if not np.all(np.isfinite(new_points)):
                alpha *= 0.5
                continue
            
            e_new, g_new = _sphere_energy_and_grad(new_points)
            
            if not np.isfinite(e_new):
                alpha *= 0.5
                continue
            
            # Armijo
            if e_new <= e0 + 1e-4 * alpha * np.sum(g0.flatten() * d.flatten()):
                found = True
                break
            
            alpha *= 0.5
        
        if not found or time.time() > deadline:
            break
        
        # Update
        g_old = g0
        current = new_points
        e0 = e_new
        g0 = g_new
        
        if e0 < best_energy:
            best_energy = e0
            best_points = current.copy()
        
        # Conjugate gradient update
        g_dot_g = np.sum(g0.flatten()**2)
        g_old_dot_g = np.sum(g_old.flatten() * g0.flatten())
        
        if g_dot_g < 1e-24:
            break
        
        beta = g_dot_g / max(g_old_dot_g, 1e-24)
        d = -g0 + beta * d
    
    return best_points, best_energy

def _lbfgs_sphere_custom(points, deadline, maxiter=500, gtol=1e-10, m=20):
    """
    Custom L-BFGS on the sphere with larger memory.
    """
    current = points.copy()
    best_points = points.copy()
    best_energy = _sphere_energy_and_grad(points)[0]
    
    if not np.isfinite(best_energy):
        return best_points, best_energy
    
    e0, g0 = _sphere_energy_and_grad(current)
    if not np.isfinite(e0):
        return best_points, best_energy
    
    # L-BFGS memory
    s_list = []
    y_list = []
    rho_list = []
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        gnorm = np.linalg.norm(g0)
        if gnorm < gtol:
            break
        
        # Compute search direction
        g_flat = g0.flatten()
        q = g_flat.copy()
        alphas = np.zeros(len(s_list))
        
        for k in range(len(s_list) - 1, -1, -1):
            s_k = s_list[k].flatten()
            y_k = y_list[k].flatten()
            rho_k = rho_list[k]
            alpha_k = rho_k * np.dot(s_k, q)
            alphas[k] = alpha_k
            q -= alpha_k * y_k
        
        if len(s_list) > 0:
            s_last = s_list[-1].flatten()
            y_last = y_list[-1].flatten()
            gamma = np.dot(s_last, y_last)
            if gamma > 1e-12:
                r = gamma / np.dot(y_last, y_last) * q
            else:
                r = q
        else:
            r = q.copy()
        
        for k in range(len(s_list)):
            s_k = s_list[k].flatten()
            y_k = y_list[k].flatten()
            rho_k = rho_list[k]
            beta_k = rho_k * np.dot(y_k, r)
            r += (alphas[k] - beta_k) * s_k
        
        direction = -r.reshape(N, 3)
        
        # Line search
        alpha = 1.0
        c1 = 1e-4
        max_ls = 25
        found_alpha = False
        
        for ls_iter in range(max_ls):
            if time.time() > deadline:
                break
            
            new_points = _retract(current, alpha * direction)
            
            if not np.all(np.isfinite(new_points)):
                alpha *= 0.5
                continue
            
            e_new, g_new = _sphere_energy_and_grad(new_points)
            
            if not np.isfinite(e_new):
                alpha *= 0.5
                continue
            
            if e_new <= e0 + c1 * alpha * np.sum(g0.flatten() * direction.flatten()):
                found_alpha = True
                break
            
            alpha *= 0.5
        
        if not found_alpha or time.time() > deadline:
            break
        
        # Update L-BFGS memory
        s = alpha * direction
        y = g_new - g0
        
        y_norm_sq = np.sum(y.flatten()**2)
        s_dot_y = np.dot(s.flatten(), y.flatten())
        
        if y_norm_sq > 1e-12 and s_dot_y > 1e-12 * y_norm_sq:
            rho = 1.0 / s_dot_y
            s_list.append(s)
            y_list.append(y)
            rho_list.append(rho)
            
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
        
        current = new_points
        e0 = e_new
        g0 = g_new
        
        if e0 < best_energy:
            best_energy = e0
            best_points = current.copy()
    
    return best_points, best_energy

def _deduplicate(points, min_dist=1e-8):
    """
    Check for and handle duplicate points.
    Returns True if points are valid (no duplicates), False otherwise.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    min_d = np.min(dists)
    if min_d < min_dist:
        # Find and separate duplicates
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        min_idx = np.argmin(d)
        i, j = iu[0][min_idx], iu[1][min_idx]
        
        # Move point j slightly
        random_dir = np.random.randn(3)
        random_dir /= np.linalg.norm(random_dir)
        points[j] = points[j] + 1e-6 * random_dir
        points[j] /= np.linalg.norm(points[j])
        
        return False
    
    return True

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    return pts / norms[:, np.newaxis]

def _spherical_harmonic_init(n, l_max=10):
    """
    Generate initial points using spherical harmonics.
    This creates a more structured initial distribution.
    """
    # Use a grid on the sphere with spherical harmonic perturbation
    n_theta = int(np.sqrt(n))
    n_phi = n // n_theta + 1
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    
    points = []
    for t in theta:
        for p in phi:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            points.append([x, y, z])
    
    points = np.array(points[:n])
    
    # Add small random perturbation
    noise = np.random.randn(n, 3) * 0.01
    points = points + noise
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points

def _optimize(points, deadline):
    """
    Multi-stage optimization:
    1. L-BFGS with moderate tolerance
    2. RCG polish for faster convergence
    3. Newton polish for quadratic convergence near minimum
    4. Ultra-polish with very tight tolerance
    """
    # Stage 1: L-BFGS
    t1 = deadline - 25.0  # Leave 25s for polish stages
    pts1, e1 = _lbfgs_sphere_custom(points, min(deadline, t1), maxiter=400, gtol=1e-10, m=20)
    
    # Stage 2: RCG polish from L-BFGS result
    t2 = deadline - 15.0
    pts2, e2 = _rcg_polish(pts1, min(deadline, t2), maxiter=200)
    
    # Stage 3: Newton polish for quadratic convergence
    t3 = deadline - 5.0
    pts3, e3 = _newton_polish(pts2, min(deadline, t3), maxiter=30)
    
    # Stage 4: Ultra-polish with very tight tolerance
    pts4, e4 = _ultra_polish(pts3, deadline, maxiter=300, gtol=1e-15)
    
    # Also try direct Newton from L-BFGS result if time permits
    if time.time() < deadline - 3.0:
        pts_n2, e_n2 = _newton_polish(pts1, deadline - 1.0, maxiter=20)
        if e_n2 < e4:
            pts4, e4 = pts_n2, e_n2
    
    return pts4, e4

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Multiple initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random points with different seeds
    for seed in [42, 123, 456, 789, 101]:
        inits.append(_random_sphere(N, seed=seed))
        
    # 3. Perturbed Fibonacci
    fib = _fibonacci_sphere(N)
    noise = np.random.randn(N, 3) * 0.05
    perturbed = fib + noise
    norms = np.linalg.norm(perturbed, axis=1)
    perturbed = perturbed / norms[:, np.newaxis]
    inits.append(perturbed)
    
    # 4. Another perturbation level
    fib2 = _fibonacci_sphere(N)
    noise2 = np.random.randn(N, 3) * 0.15
    perturbed2 = fib2 + noise2
    norms2 = np.linalg.norm(perturbed2, axis=1)
    perturbed2 = perturbed2 / norms2[:, np.newaxis]
    inits.append(perturbed2)
    
    # 5. Spherical harmonic init
    inits.append(_spherical_harmonic_init(N))
    
    best_points = None
    best_energy = float('inf')
    
    # Sort inits by initial energy to try good ones first
    init_energies = []
    for init in inits:
        try:
            e, _ = _sphere_energy_and_grad(init)
            if np.isfinite(e):
                init_energies.append((e, init))
        except:
            init_energies.append((float('inf'), init))
    
    init_energies.sort(key=lambda x: x[0])
    sorted_inits = [x[1] for x in init_energies]
    
    for i, init in enumerate(sorted_inits):
        # Check time
        remaining = deadline - time.time()
        if remaining < 20.0:  # Need at least 20s for a full optimization
            break
            
        # Check for duplicates
        if not _deduplicate(init):
            continue
            
        # Optimize
        opt_points, energy = _optimize(init, deadline)
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
            
    if best_points is None:
        return _fibonacci_sphere(N)
        
    # Final validation
    final_e, _ = _sphere_energy_and_grad(best_points)
    if not np.isfinite(final_e):
        return _fibonacci_sphere(N)
        
    return best_points
# EVOLVE-BLOCK-END
