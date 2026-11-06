"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    if n < 2:
        wall = 0.5 - np.max(np.abs(centers), axis=1)
        return np.min(wall)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _soft_radius_and_grad(centers, beta):
    """
    Compute the negative soft-min radius (to be minimized) and its gradient.
    Objective: -soft_min(dists)
    soft_min(dists) = max_d - (1/beta) * log(sum(exp(-beta * (d_i - max_d))))
    We want to maximize soft_min, so we minimize -soft_min.
    """
    n = len(centers)
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    dists_pair = np.sqrt(d2) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    
    # Flatten
    pair_flat = dists_pair[np.triu_indices(n, k=1)]
    wall_flat = wall_dists.flatten()
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    # Soft min calculation
    max_d = np.max(all_dists)
    # Numerical stability: subtract max_d
    shifted = all_dists - max_d
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    
    soft_min_val = max_d - log_sum / beta
    obj_val = -soft_min_val
    
    # Gradient
    # d(soft_min)/dx = - (1/sum_exp) * sum_i exp(-beta*d_i) * d(d_i)/dx
    # We want gradient of -soft_min, so:
    # grad_obj = (1/sum_exp) * sum_i exp(-beta*d_i) * d(d_i)/dx
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Pairwise part
    # d(d_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # Note: d_ij = |c_i - c_j| / 2, so |c_i - c_j| = 2 * d_ij
    # d(d_ij)/d(c_i) = (c_i - c_j) / (2 * 2 * d_ij) = (c_i - c_j) / (4 * d_ij)
    # Wait, d_ij = 0.5 * |c_i - c_j|. 
    # Let r_ij = |c_i - c_j|. d_ij = r_ij / 2.
    # d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / r_ij.
    
    abs_diff = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero
    abs_diff[abs_diff == 0] = 1e-12
    
    # Unit vectors
    dir_vec = diff / abs_diff[:, :, np.newaxis]
    
    # Gradient contribution from pairs
    # w_ij is weight for pair (i,j)
    # grad[i] += sum_j w_ij * 0.5 * dir_vec[i,j]
    # grad[j] += sum_i w_ij * 0.5 * dir_vec[j,i] = sum_i w_ij * 0.5 * (-dir_vec[i,j])
    
    pair_weights = weights[:len(pair_flat)]
    # Reconstruct weight matrix
    pair_w_matrix = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    pair_w_matrix[iu] = pair_weights
    pair_w_matrix = pair_w_matrix + pair_w_matrix.T
    
    # Weighted sum of directions
    # weighted_dir[i, j] = w_ij * dir_vec[i, j]
    weighted_dir = pair_w_matrix[:, :, np.newaxis] * dir_vec
    pair_grad = np.sum(weighted_dir, axis=1) * 0.5
    
    # Wall part
    # d(d_ik)/d(c_ik) = -sign(c_ik)
    wall_weights = weights[len(pair_flat):]
    wall_w_matrix = wall_weights.reshape(n, 3)
    
    signs = np.sign(centers)
    wall_grad = -wall_w_matrix * signs
    
    # Total gradient of -soft_min
    grad = pair_grad + wall_grad
    
    return obj_val, grad


def _bcc_init(n_points):
    """Generate initial points on a distorted BCC lattice."""
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                
                points.append([ox, oy, oz])
                points.append([ox + 1.0/6.0, oy + 1.0/6.0, oz + 1.0/6.0])
                
    points = np.array(points)
    return points[:n_points].copy()


def _fcc_init(n_points):
    """Generate initial points on an FCC-like lattice."""
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                cell = 1.0/3.0
                
                points.append([ox, oy, oz])
                points.append([ox + cell/2, oy + cell/2, oz])
                points.append([ox + cell/2, oy, oz + cell/2])
                points.append([ox, oy + cell/2, oz + cell/2])
    
    points = np.array(points)
    return points[:n_points].copy()


def _random_init(n_points, seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n_points, 3))


def _optimize_smooth(centers, time_budget):
    """
    Optimize using smooth surrogate with increasing beta.
    """
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    
    best_r = _radius(centers)
    best_centers = centers.copy()
    
    # Schedule of betas: start smooth, end sharp
    betas = [50, 200, 1000, 5000]
    
    for beta in betas:
        if time.time() - start > time_budget:
            break
            
        def objective(x):
            c = x.reshape(n, 3)
            c = np.clip(c, -0.5, 0.5)
            val, _ = _soft_radius_and_grad(c, beta)
            return val
        
        def gradient(x):
            c = x.reshape(n, 3)
            c = np.clip(c, -0.5, 0.5)
            _, grad = _soft_radius_and_grad(c, beta)
            return grad.flatten()
        
        x0 = centers.flatten()
        
        try:
            result = opt.minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  bounds=[(-0.5, 0.5)] * (n * 3),
                                  options={'maxiter': 100, 'maxls': 50})
            
            if result.success or result.fun < objective(x0):
                centers = result.x.reshape(n, 3)
                centers = np.clip(centers, -0.5, 0.5)
                
                current_r = _radius(centers)
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                    
        except Exception:
            pass
            
    return best_centers, best_r


def _hard_polish(centers, time_budget):
    """
    Final polish using Nelder-Mead on the hard radius.
    """
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    best_r = _radius(centers)
    best_centers = centers.copy()
    
    def objective(x):
        c = x.reshape(n, 3)
        c = np.clip(c, -0.5, 0.5)
        return -_radius(c)
    
    x0 = centers.flatten()
    
    try:
        # Nelder-Mead is derivative-free and robust for final polishing
        # We run it for a short time
        result = opt.minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if result.fun < -best_r:
            centers = result.x.reshape(n, 3)
            centers = np.clip(centers, -0.5, 0.5)
            current_r = _radius(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
                
    except Exception:
        pass
        
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # List of initializers
    initializers = [
        _bcc_init(N),
        _fcc_init(N),
        _random_init(N, 1),
        _random_init(N, 2),
        _random_init(N, 3)
    ]
    
    best_centers = None
    best_r = -np.inf
    
    # Allocate time per initializer
    # We have ~140s. 5 initializers. ~25s each?
    # But smooth optimization is fast. Hard polish is slower.
    # Let's do 2s per beta step for smooth, 5s for hard polish per start.
    # Total per start: ~13s. 5 starts = 65s. Plenty of room.
    
    for init_centers in initializers:
        if time.time() - start_time > deadline - 10:
            break
            
        # Smooth optimization
        budget_smooth = 10.0
        c_smooth, r_smooth = _optimize_smooth(init_centers, budget_smooth)
        
        # Hard polish
        budget_hard = 5.0
        c_final, r_final = _hard_polish(c_smooth, budget_hard)
        
        if r_final > best_r:
            best_r = r_final
            best_centers = c_final.copy()
            
        # Check time
        if time.time() - start_time > deadline - 5:
            break
            
    # If no valid result, fallback to grid
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        
    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        
    return best_centers
# EVOLVE-BLOCK-END
