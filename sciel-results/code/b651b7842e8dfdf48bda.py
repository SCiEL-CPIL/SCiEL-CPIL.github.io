"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_spatial_hash(points, cell_size):
    """Build a spatial hash for fast neighbor lookup."""
    n = len(points)
    cell = {}
    for i in range(n):
        cx = int(np.floor(points[i, 0] / cell_size))
        cy = int(np.floor(points[i, 1] / cell_size))
        key = (cx, cy)
        if key not in cell:
            cell[key] = []
        cell[key].append(i)
    return cell

def _get_neighbors_spatial(cell, point, cell_size, max_dist_sq):
    """Get indices of points within max_dist_sq of point using spatial hash."""
    cx = int(np.floor(point[0] / cell_size))
    cy = int(np.floor(point[1] / cell_size))
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            key = (cx + dx, cy + dy)
            if key in cell:
                for idx in cell[key]:
                    d2 = np.sum((point - None) ** 2)  # placeholder
                    # We'll compute actual distance in the caller
                    neighbors.append(idx)
    return neighbors

def _compute_forces_spatial(points, k_soft, cell_size=None):
    """Compute forces using spatial hashing for O(N) complexity."""
    n = len(points)
    if cell_size is None:
        # Estimate cell size based on current spacing
        # Use a reasonable default
        cell_size = 0.1
    
    # Build spatial hash
    cell = _build_spatial_hash(points, cell_size)
    
    # Compute all constraint values first
    # For efficiency, we'll compute pairwise distances only for nearby pairs
    # and boundary distances for all points
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    current_min = np.min(bound_dists)
    
    # Find minimum pairwise distance using spatial hash
    min_pair_dist = np.inf
    min_pair_i = 0
    min_pair_j = 0
    
    # For each point, check neighbors
    for i in range(n):
        cx = int(np.floor(points[i, 0] / cell_size))
        cy = int(np.floor(points[i, 1] / cell_size))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in cell:
                    for j in cell[key]:
                        if j > i:  # Avoid double counting
                            d = np.linalg.norm(points[i] - points[j])
                            if d < min_pair_dist:
                                min_pair_dist = d
                                min_pair_i = i
                                min_pair_j = j
    
    if min_pair_dist < current_min:
        current_min = min_pair_dist / 2.0
    
    # Compute weights for all constraints
    # Boundary weights
    shifted_bound = bound_dists - current_min
    bound_weights = np.exp(-k_soft * shifted_bound)
    bound_sum = np.sum(bound_weights)
    if bound_sum > 0:
        bound_weights /= bound_sum
    else:
        bound_weights = np.ones(n) / n
    
    # Pair weights - compute for nearby pairs only
    pair_weights = {}
    for i in range(n):
        cx = int(np.floor(points[i, 0] / cell_size))
        cy = int(np.floor(points[i, 1] / cell_size))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in cell:
                    for j in cell[key]:
                        if j > i:
                            d = np.linalg.norm(points[i] - points[j])
                            val = d / 2.0
                            shifted = val - current_min
                            if shifted < 50.0:  # Only consider close constraints
                                w = np.exp(-k_soft * shifted)
                                if w > 1e-10:
                                    pair_weights[(i, j)] = w
    
    # Normalize pair weights
    total_pair_weight = sum(pair_weights.values())
    if total_pair_weight > 0:
        for key in pair_weights:
            pair_weights[key] /= total_pair_weight
    else:
        # If no pair weights, distribute evenly
        pair_count = len(pair_weights)
        if pair_count > 0:
            for key in pair_weights:
                pair_weights[key] = 1.0 / pair_count
        else:
            pair_weights = {}
    
    # Compute gradients
    grads = np.zeros_like(points)
    
    # Boundary contributions
    rad_safe = np.where(rad > 1e-9, rad, 1.0)
    grads -= (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
    
    # Pair contributions
    for (i, j), w in pair_weights.items():
        diff_ij = points[i] - points[j]
        d = np.linalg.norm(diff_ij)
        if d > 1e-9:
            force_vec = (diff_ij / d) * w
            grads[i] += force_vec
            grads[j] -= force_vec
    
    return grads, current_min

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0
    
    # Precompute indices for pairwise distances (for final polish)
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    # Phase 1: Spatial-hash accelerated optimization
    alpha = 0.015
    cooling = 0.99995
    k_soft_start = 10.0
    k_soft_end = 100.0
    cell_size = 0.08
    iter_count = 0
    stall_count = 0
    stall_threshold = 50
    max_iter_phase1 = 50000
    
    k_soft = k_soft_start
    
    while time.time() - start_time < time_limit * 0.6 and iter_count < max_iter_phase1:
        # Compute forces using spatial hashing
        grads, current_min = _compute_forces_spatial(points, k_soft, cell_size)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        # Update points
        points += alpha * grads
        
        # Project back inside circle
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Cool down
        alpha *= cooling
        
        # Increase k_soft over time
        progress = iter_count / max_iter_phase1
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 1.5)
        
        # Random perturbation to escape local minima
        if stall_count > stall_threshold:
            noise_scale = alpha * 2.0
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count = 0
            alpha *= 0.9
            
        iter_count += 1

    # Phase 2: Full pairwise optimization with tighter parameters
    points = best_points.copy()
    alpha2 = 0.005
    cooling2 = 0.9999
    k_soft2_start = 50.0
    k_soft2_end = 500.0
    max_iter_phase2 = 20000
    iter_count2 = 0
    stall_count2 = 0
    stall_threshold2 = 30
    
    k_soft2 = k_soft2_start
    
    while time.time() - start_time < time_limit * 0.85 and iter_count2 < max_iter_phase2:
        # Full pairwise computation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            stall_count2 = 0
        else:
            stall_count2 += 1
        
        # Soft-min weights
        shifted_vals = all_vals - current_min
        mask_close = shifted_vals < 30.0
        weights = np.zeros_like(all_vals)
        if np.any(mask_close):
            close_vals = shifted_vals[mask_close]
            close_weights = np.exp(-k_soft2 * close_vals)
            weights[mask_close] = close_weights
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)
        
        # Compute gradients
        grads = np.zeros_like(points)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += alpha2 * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        alpha2 *= cooling2
        
        progress2 = iter_count2 / max_iter_phase2
        k_soft2 = k_soft2_start + (k_soft2_end - k_soft2_start) * (progress2 ** 2)
        
        if stall_count2 > stall_threshold2:
            noise_scale = alpha2 * 2.5
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count2 = 0
            alpha2 *= 0.9
            
        iter_count2 += 1

    # Phase 3: High-precision polish with very tight tolerance
    points = best_points.copy()
    polish_alpha = 0.0005
    polish_k = 2000.0
    polish_time_limit = time_limit * 0.98
    polish_iter = 0
    max_polish_iter = 10000
    polish_stall = 0
    polish_stall_threshold = 20
    
    while time.time() - start_time < polish_time_limit and polish_iter < max_polish_iter:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            polish_stall = 0
        else:
            polish_stall += 1
        
        # Very sharp soft-min
        shifted_vals = all_vals - current_min
        mask_close = shifted_vals < 10.0
        weights = np.zeros_like(all_vals)
        if np.any(mask_close):
            close_vals = shifted_vals[mask_close]
            close_weights = np.exp(-polish_k * close_vals)
            weights[mask_close] = close_weights
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)
        
        grads = np.zeros_like(points)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += polish_alpha * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        polish_alpha *= 0.9995
        if polish_alpha < 1e-7:
            break
        
        if polish_stall > polish_stall_threshold:
            # Small perturbation
            points += np.random.normal(0, 1e-5, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            polish_stall = 0
        
        polish_iter += 1

    return best_points

# EVOLVE-BLOCK-END
