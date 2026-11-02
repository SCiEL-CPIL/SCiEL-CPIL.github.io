import numpy as np
from scipy.optimize import minimize
import time
import heapq

N = 300

def _hex_lattice_init(n_points, rotation=0.0, noise=0.0):
    """Generate a hexagonal lattice of points inside the unit circle."""
    # Estimate spacing
    # Area of circle = pi. Area per point = pi / n.
    # Hex lattice area per point = s * s * sqrt(3)/2.
    # s^2 * sqrt(3)/2 = pi / n => s = sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if rotation != 0.0:
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        points = points @ R.T
        
    # Add noise if requested
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
        
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    # Ensure they are inside
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        points[mask] = points[mask] / norms[mask, np.newaxis] * 0.99
        
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    return min(min_pair, min_wall)

def _sparse_obj_and_grad(x, K, beta):
    """
    Compute negative soft-min objective and gradient using only K nearest neighbors.
    x: shape (2*N,)
    """
    C = x.reshape(N, 2)
    
    # Compute pairwise differences and distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    
    # Find K smallest distances
    # Flatten L, find K smallest
    L_flat = L.flatten()
    # np.argpartition is O(N)
    part_indices = np.argpartition(L_flat, K)[:K]
    
    # Get the actual values and indices
    # We need to map flat indices back to (i, j)
    i_idx, j_idx = np.unravel_index(part_indices, (N, N))
    
    # Filter out self-interactions just in case (should be inf)
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    L_subset = L_flat[part_indices][mask]
    
    # Ensure L_subset is positive
    L_subset = np.maximum(L_subset, 1e-12)
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    
    # Combine pair distances (halved) and wall distances
    x_pairs = L_subset / 2.0
    x_walls = 1.0 - L_wall
    
    x_all = np.concatenate([x_pairs, x_walls])
    
    # Soft-min
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    val = min_x - (1.0 / beta) * np.log(Z)
    
    # Gradient
    grad = np.zeros_like(C)
    
    # Pair gradients
    P_pairs = P[:len(i_idx)]
    D_subset = D[i_idx, j_idx]
    L_subset_safe = np.where(L_subset > 1e-12, L_subset, 1.0)
    U = D_subset / L_subset_safe[:, np.newaxis]
    
    # Gradient of d_ij/2 wrt c_i is 0.5 * (c_i - c_j) / d_ij
    # Gradient of d_ij/2 wrt c_j is -0.5 * (c_i - c_j) / d_ij
    # We are minimizing -val, so gradient is -d(val)/dx
    # val = softmin. d(softmin)/dx = sum P_k * d(x_k)/dx
    # So d(-val)/dx = - sum P_k * d(x_k)/dx
    
    # We want to maximize val, so minimize -val.
    # L-BFGS minimizes.
    # So we return -val and -grad_val.
    
    # grad_val for pairs:
    # d(x_pair_k)/d(c_i) = 0.5 * U_k
    # d(x_pair_k)/d(c_j) = -0.5 * U_k
    # Contribution to grad_val at i: sum_{k where i is in pair} P_k * 0.5 * U_k
    # Contribution to grad_val at j: sum_{k where j is in pair} P_k * (-0.5 * U_k)
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    # Wall gradients
    P_walls = P[len(i_idx):]
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    # x_wall = 1 - |c|. d(x_wall)/d(c) = -c / |c|
    U_walls = -C / L_wall_safe[:, np.newaxis]
    grad += P_walls[:, np.newaxis] * U_walls
    
    # Return -val and -grad (since we are minimizing -val)
    return -val, -grad

def _optimize_with_sparse_lbfgs(points_init, time_limit):
    """
    Optimize using sparse soft-min gradient with L-BFGS-B.
    """
    start_time = time.time()
    best_points = points_init.copy()
    best_score = _compute_true_score(points_init)
    current_points = points_init.copy()
    
    # Annealing schedule for K and beta
    # Start with large K (more global) and low beta (smooth)
    # End with small K (local) and high beta (sharp)
    
    stages = [
        {'K': 100, 'beta': 50, 'max_iter': 200},
        {'K': 50, 'beta': 200, 'max_iter': 200},
        {'K': 20, 'beta': 1000, 'max_iter': 300},
        {'K': 10, 'beta': 5000, 'max_iter': 500},
    ]
    
    for stage in stages:
        if time.time() - start_time > time_limit - 5:
            break
            
        K = stage['K']
        beta = stage['beta']
        max_iter = stage['max_iter']
        
        def obj_grad(x):
            return _sparse_obj_and_grad(x, K, beta)
        
        try:
            res = minimize(obj_grad, current_points.flatten(), method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
            
            new_points = res.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_points)):
                continue
                
            # Check validity
            norms = np.linalg.norm(new_points, axis=1)
            if np.max(norms) > 1.0:
                # Scale back
                scale = 0.999 / np.max(norms)
                new_points = new_points * scale
            
            new_score = _compute_true_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
            # Update current points for next stage if it's valid
            if np.all(np.isfinite(new_points)):
                current_points = new_points.copy()
                
        except Exception:
            pass
            
    return best_points, best_score

def _local_relaxation(points, steps=500, time_limit=10.0):
    """
    Simple force-directed relaxation to escape local minima.
    """
    start_time = time.time()
    C = points.copy()
    dt = 0.01
    
    for s in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        L = np.linalg.norm(D, axis=2)
        np.fill_diagonal(L, np.inf)
        
        # Current min distance
        min_d = np.min(L)
        if min_d < 1e-6:
            # Handle overlaps
            # Add small random jitter
            C += np.random.normal(0, 0.01, C.shape)
            continue
            
        # Target radius is roughly min_d / 2
        # We want to push apart pairs that are close to min_d
        # Force F_ij = (min_d - L_ij) * (C_i - C_j) / L_ij if L_ij < min_d * 1.2
        # This is a bit crude. Let's use a soft repulsion.
        
        # Repulsion kernel
        # k = 1 / (1 + (L / min_d)^4)
        # This is 1 when L << min_d, 0 when L >> min_d.
        # Actually we want to push apart when L is small.
        
        # Let's use the gradient of the sum of squared overlaps?
        # Or just simple repulsion for L < threshold.
        
        threshold = min_d * 1.5
        mask = L < threshold
        # Exclude diagonal
        mask[np.diag_indices(N)] = False
        
        # Force
        F = np.zeros_like(C)
        # Only compute for masked pairs to save time? 
        # N=300, N^2=90k. It's fine to compute all.
        
        L_safe = np.where(L > 1e-6, L, 1.0)
        # Weight: stronger for smaller L
        W = np.where(mask, (threshold - L) / threshold, 0.0)
        W = W / L_safe[:, :, np.newaxis]
        
        F = np.sum(W * D, axis=1)
        
        # Boundary force
        norms = np.linalg.norm(C, axis=1)
        # Push in if close to boundary
        # F_boundary = -C / norm * (norm - (1 - r_est))
        r_est = min_d / 2
        dist_to_wall = 1.0 - norms
        wall_force = np.where(dist_to_wall < r_est, (r_est - dist_to_wall) * (-C / norms[:, np.newaxis]), 0.0)
        
        F += wall_force
        
        # Update
        C += dt * F
        
        # Keep inside
        norms = np.linalg.norm(C, axis=1)
        mask_out = norms > 0.99
        if np.any(mask_out):
            C[mask_out] = C[mask_out] / norms[mask_out, np.newaxis] * 0.99
            
    return C

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    inits = [
        lambda: _hex_lattice_init(N, rotation=0.0, noise=0.0),
        lambda: _hex_lattice_init(N, rotation=np.pi/12, noise=0.0),
        lambda: _hex_lattice_init(N, rotation=0.0, noise=0.005),
        lambda: _hex_lattice_init(N, rotation=np.pi/6, noise=0.005),
    ]
    
    # Allocate time per init
    # 4 inits. 30s each? Total 120s.
    # Plus final polish.
    
    for i, init_func in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            break
            
        # Give each init about 30s
        slice_time = min(30.0, time_limit - elapsed - 5.0)
        if slice_time < 5.0:
            break
            
        try:
            points_init = init_func()
            
            # Phase 1: Local Relaxation (fast, escapes bad local minima)
            # Use a small time slice for this
            relax_time = min(5.0, slice_time * 0.2)
            points_relaxed = _local_relaxation(points_init, steps=200, time_limit=relax_time)
            
            # Phase 2: Sparse L-BFGS Polish
            polish_time = slice_time - relax_time - 1.0
            if polish_time > 2.0:
                points_opt, score_opt = _optimize_with_sparse_lbfgs(points_relaxed, polish_time)
            else:
                points_opt = points_relaxed
                score_opt = _compute_true_score(points_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt.copy()
                
        except Exception:
            continue
            
    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_true_score(best_points)
        
    # Final Polish with remaining time
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5.0:
        remaining = time_limit - elapsed - 2.0
        # Use a very sharp beta and small K for final precision
        # We can just call the sparse optimizer again with tighter settings
        # Or a dedicated final polish function
        
        # Let's do one more pass of sparse L-BFGS with high beta
        def final_obj_grad(x):
            return _sparse_obj_and_grad(x, K=5, beta=10000)
            
        try:
            res = minimize(final_obj_grad, best_points.flatten(), method='L-BFGS-B', jac=True,
                          options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15})
            new_points = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0:
                    new_score = _compute_true_score(new_points)
                    if new_score > best_score:
                        best_points = new_points
                        best_score = new_score
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        best_points = _hex_lattice_init(N)
        
    dists = np.linalg.norm(best_points, axis=1)
    if np.max(dists) > 1.0:
        scale = 0.999 / np.max(dists)
        best_points = best_points * scale
        
    return best_points
