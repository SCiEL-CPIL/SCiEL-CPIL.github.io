import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _energy_and_grad(x):
    """
    x: shape (N, 3) on unit sphere
    Returns energy and gradient in R^3 (not projected).
    Energy: sum_{i<j} log(1/||xi - xj||)
    Gradient: dE/dx_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    energy = -0.5 * np.sum(np.log(dist[mask]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    return energy, grad

def _hessian(x):
    """
    Compute the Hessian of the energy function.
    Returns a (3N, 3N) matrix.
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)
    
    # H_kl for k != l:
    # H_kl = -I / r_kl^2 + 2 * outer(diff_kl, diff_kl) / r_kl^4
    # H_kk = sum_{j != k} [ I / r_kj^2 - 2 * outer(diff_kj, diff_kj) / r_kj^4 ]
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    inv_dist_sq2 = inv_dist_sq ** 2
    np.fill_diagonal(inv_dist_sq2, 0.0)
    
    # Term 1: -I / r^2
    # This contributes to the diagonal of the block matrix
    # For off-diagonal blocks (k != l), it is -inv_dist_sq[k,l] * I
    
    # Term 2: 2 * outer(diff, diff) / r^4
    # diff is (N, N, 3)
    # outer_diff is (N, N, 3, 3)
    outer_diff = diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :]
    # Scale by 2 * inv_dist_sq2
    term2 = 2.0 * inv_dist_sq2[:, :, np.newaxis, np.newaxis] * outer_diff
    
    # Construct Hessian blocks
    # H[k, l] is a 3x3 matrix
    # H[k, l] = -inv_dist_sq[k,l] * I + term2[k,l]  for k != l
    # H[k, k] = sum_{j != k} ( inv_dist_sq[k,j] * I - term2[k,j] )
    
    H_blocks = np.zeros((N, N, 3, 3))
    eye = np.eye(3)
    
    # Off-diagonal blocks
    for k in range(N):
        for l in range(N):
            if k != l:
                H_blocks[k, l] = -inv_dist_sq[k, l] * eye + term2[k, l]
    
    # Diagonal blocks
    for k in range(N):
        H_diag = np.zeros((3, 3))
        for j in range(N):
            if k != j:
                H_diag += inv_dist_sq[k, j] * eye - term2[k, j]
        H_blocks[k, k] = H_diag
        
    # Flatten to (3N, 3N)
    H = np.zeros((3*N, 3*N))
    for k in range(N):
        for l in range(N):
            H[k*3:(k+1)*3, l*3:(l+1)*3] = H_blocks[k, l]
            
    return H

def _tangent_project(x, v):
    """
    Project vector v onto tangent space at x.
    x, v: shape (N, 3)
    """
    vx = np.einsum('ij,ij->i', v, x)
    return v - vx[:, np.newaxis] * x

def _retract(x, step):
    """Retract x + step back to sphere."""
    new_x = x + step
    norms = np.linalg.norm(new_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return new_x / norms

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def _random_sphere(seed):
    rng = np.random.RandomState(seed)
    x = rng.randn(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms

def _optimize_newton(x0, max_iter=50, deadline=None):
    """
    Newton optimization with tangent space projection and retraction.
    """
    x = x0.copy()
    best_x = x0.copy()
    best_e, _ = _energy_and_grad(x)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(x)
        if e < best_e:
            best_e = e
            best_x = x.copy()
            
        g_tan = _tangent_project(x, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        # Compute Hessian
        H = _hessian(x)
        
        # Project Hessian to tangent space
        # H_tan = P H P
        # P is block diagonal with P_k = I - x_k x_k^T
        H_tan = np.zeros_like(H)
        for k in range(N):
            P_k = np.eye(3) - np.outer(x[k], x[k])
            for l in range(N):
                H_tan[k*3:(k+1)*3, l*3:(l+1)*3] = P_k @ H[k*3:(k+1)*3, l*3:(l+1)*3] @ (np.eye(3) - np.outer(x[l], x[l]))
                
        # Solve H_tan * step = -g_tan
        # Add small regularization to ensure positive definiteness
        # H_tan might be singular or ill-conditioned
        try:
            step = np.linalg.solve(H_tan + 1e-8 * np.eye(3*N), -g_tan.flatten())
        except np.linalg.LinAlgError:
            step = -g_tan.flatten()
            
        step = step.reshape(N, 3)
        
        # Line search
        alpha = 1.0
        for ls_it in range(30):
            new_x = _retract(x, alpha * step)
            new_e, _ = _energy_and_grad(new_x)
            
            # Armijo condition
            if new_e <= e + 1e-4 * alpha * np.dot(g_tan.flatten(), step.flatten()):
                break
            alpha *= 0.5
            
        if alpha < 1e-10:
            break
            
        x = new_x
        
    return best_x, best_e

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate initial guesses
    inits = []
    inits.append(_fibonacci_spiral())
    
    # Add a few random inits
    for seed in [1, 2, 3]:
        inits.append(_random_sphere(seed))
        
    best_x = None
    best_e = float('inf')
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        x_opt, e_opt = _optimize_newton(x0, max_iter=50, deadline=deadline)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
            
    if best_x is None:
        best_x = _fibonacci_spiral()
        
    # Final normalization
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_x = best_x / norms
    
    # Final check: ensure no points are too close
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Perturb the closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        best_x[i] += 1e-8 * np.random.RandomState(123).randn(3)
        best_x[i] = best_x[i] / np.linalg.norm(best_x[i])
        
    return best_x
