import numpy as np
import time
from scipy.optimize import minimize
import random

N = 50

def _mobius_transform(points, a, theta):
    """
    Apply a Möbius transformation to points in the unit disk.
    T(z) = exp(i*theta) * (z - a) / (1 - conj(a)*z)
    points: (N, 2) array of [x, y]
    a: complex number, |a| < 1
    theta: float
    """
    z = points[:, 0] + 1j * points[:, 1]
    a_c = a
    denom = 1 - np.conj(a_c) * z
    # Avoid division by zero, though |a|<1 and |z|<1 ensures denom != 0
    w = np.exp(1j * theta) * (z - a_c) / denom
    return np.column_stack((w.real, w.imag))

def _score(points):
    """
    Compute the score: min(min_pair_dist/2, min_boundary_dist)
    """
    if len(points) != N:
        return -1.0
    # Check if points are in disk
    dists_origin = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - dists_origin
    if np.any(boundary_dists < -1e-9):
        return -1.0
        
    # Pairwise distances
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, np.min(boundary_dists))
    return r

def _smooth_score(points, beta):
    """
    Smooth surrogate for the score.
    We want to maximize min(v_k).
    min(v_k) approx -1/beta * log(sum(exp(-beta * v_k)))
    So we minimize 1/beta * log(sum(exp(-beta * v_k)))
    """
    # Boundary terms
    dists_origin = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Pairwise terms
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Flatten all values
    # To avoid memory explosion for N=50, we can compute in chunks or just use the full matrix (50x50 is small)
    vals = np.concatenate([boundary_dists, pair_dists.ravel()])
    
    # Log-sum-exp trick for stability
    m = np.min(vals)
    if m < -100: # Very bad
        return 100.0
    exp_vals = np.exp(-beta * (vals - m))
    log_sum_exp = m + np.log(np.sum(exp_vals))
    return -log_sum_exp / beta

def _grad_smooth_score(points, beta):
    """
    Gradient of the smooth score.
    """
    dists_origin = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    vals = np.concatenate([boundary_dists, pair_dists.ravel()])
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    
    # d/dv_k (-1/beta * log(sum(exp(-beta*v)))) = -1/beta * (1/sum) * exp(-beta*v_k) * (-beta) = exp(-beta*v_k)/sum
    # So grad_obj = sum_k (exp(-beta*v_k)/sum) * grad_v_k
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(points)
    
    # Boundary part: v_i = 1 - ||p_i||
    # d v_i / d p_i = - p_i / ||p_i||
    w_b = weights[:N]
    # Handle p_i = 0
    norms = dists_origin
    norms_safe = np.where(norms < 1e-8, 1.0, norms)
    grad -= w_b[:, np.newaxis] * (points / norms_safe[:, np.newaxis])
    
    # Pairwise part: v_ij = ||p_i - p_j|| / 2
    # d v_ij / d p_i = (p_i - p_j) / (2 * ||p_i - p_j||)
    # d v_ij / d p_j = (p_j - p_i) / (2 * ||p_i - p_j||)
    
    # weights[N:] are for pairs. We need to map them back to indices.
    # The order in pair_dists.ravel() is row-major.
    # Index k in ravel corresponds to i = k // N, j = k % N.
    w_p = weights[N:]
    
    # We can vectorize this
    # Create a matrix of weights W[i,j] = w_p[i*N+j]
    W = w_p.reshape(N, N)
    
    # Diff matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    # Avoid div by 0
    dists_safe = np.where(dists_matrix < 1e-8, 1.0, dists_matrix)
    
    # grad_i = sum_j W[i,j] * (p_i - p_j) / (2 * dist_ij)
    # We can compute this by summing over j for each i
    # term_ij = W[i,j] * diff[i,j] / (2 * dists_safe[i,j])
    term = W[:, :, np.newaxis] * diff / (2 * dists_safe[:, :, np.newaxis])
    grad += np.sum(term, axis=1)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    spacing = 0.2
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 <= 0.9**2:
                pts.append([x, y])
    if len(pts) >= N:
        inits.append(np.array(pts[:N]))
    else:
        inits.append(np.array(pts + [[0.1, 0.1]] * (N - len(pts))))

    # 2. Concentric rings
    pts = []
    rings = [0.0, 0.3, 0.6, 0.9]
    for r in rings:
        if r == 0:
            pts.append([0,0])
        else:
            n = int(2 * np.pi * r / 0.2)
            for k in range(n):
                theta = 2 * np.pi * k / n
                pts.append([r * np.cos(theta), r * np.sin(theta)])
    if len(pts) < N:
        # Add random
        while len(pts) < N:
            x, y = np.random.uniform(-0.9, 0.9, 2)
            if x**2 + y**2 < 0.81:
                pts.append(list(x, y))
    inits.append(np.array(pts[:N]))

    # 3. Random
    while len(inits) < 3:
        pts = np.random.uniform(-0.9, 0.9, (N, 2))
        # Ensure inside
        norms = np.linalg.norm(pts, axis=1)
        pts[norms > 0.9] *= 0.9 / norms[norms > 0.9][:, np.newaxis]
        inits.append(pts)

    betas = [10, 50, 100, 500, 1000, 5000, 10000]
    
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        current_points = init_points.copy()
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            # Optimize
            try:
                res = minimize(
                    lambda x: _smooth_score(x.reshape(N, 2), beta),
                    current_points.flatten(),
                    jac=lambda x: _grad_smooth_score(x.reshape(N, 2), beta).flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                current_points = res.x.reshape(N, 2)
            except Exception as e:
                pass
            
            # Check score
            s = _score(current_points)
            if s > best_score:
                best_score = s
                best_points = current_points.copy()
                
            # Möbius Perturbation
            if time.time() < deadline - 5:
                # Apply a random Möbius transformation
                a_mag = np.random.uniform(0, 0.5)
                a_ang = np.random.uniform(0, 2*np.pi)
                a = a_mag * np.exp(1j * a_ang)
                theta = np.random.uniform(0, 2*np.pi)
                
                transformed = _mobius_transform(current_points, a, theta)
                
                # Check if transformed is better
                s_t = _score(transformed)
                if s_t > s:
                    current_points = transformed
                    best_score = max(best_score, s_t)
                    if best_points is None or s_t > _score(best_points):
                        best_points = transformed.copy()

    # Final polish with high beta
    if best_points is not None and time.time() < deadline - 5:
        for beta in [50000, 100000]:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(
                    lambda x: _smooth_score(x.reshape(N, 2), beta),
                    best_points.flatten(),
                    jac=lambda x: _grad_smooth_score(x.reshape(N, 2), beta).flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-14}
                )
                candidate = res.x.reshape(N, 2)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
            except:
                pass

    if best_points is None:
        return inits[0]
        
    return best_points

# EVOLVE-BLOCK-END
