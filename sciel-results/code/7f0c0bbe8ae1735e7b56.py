import numpy as np
import time
from scipy.spatial import Delaunay
import scipy.sparse as sp

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Target spacing for N=300 in unit circle.
    # N ~ 2*pi / (sqrt(3)*s^2) => s ~ sqrt(2*pi / (sqrt(3)*N))
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    
    # Generate a slightly denser lattice to have more points, then prune
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
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def _score(pts):
        """Compute the packing score."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists[i_idx, j_idx]) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_min = np.min(1.0 - rad)
        return min(pair_min, bound_min)
    
    def _soft_min_weights(vals, k_soft):
        """Compute soft-min weights with numerical stability."""
        current_min = np.min(vals)
        shifted = vals - current_min
        
        # For numerical stability, only consider values within a reasonable range
        threshold = 50.0 / k_soft
        mask = shifted < threshold
        
        if np.any(mask):
            close_shifted = shifted[mask]
            # Use log-sum-exp for stability
            max_close = np.max(close_shifted)
            if max_close > 0:
                exp_terms = np.exp(-k_soft * (close_shifted - max_close))
            else:
                exp_terms = np.exp(-k_soft * close_shifted)
            
            sum_exp = np.sum(exp_terms)
            if sum_exp > 0:
                weights = np.zeros_like(vals)
                weights[mask] = exp_terms / sum_exp
            else:
                weights = np.ones_like(vals) / len(vals)
        else:
            weights = np.ones_like(vals) / len(vals)
        
        return weights, current_min

    def _spectral_regularization_grad(pts, weight=0.1):
        """
        Computes gradient of the variance of Delaunay edge lengths.
        Minimizing this variance promotes a more uniform lattice structure,
        acting as a global regularizer to enforce rigidity and uniformity.
        """
        try:
            # Compute Delaunay triangulation
            tri = Delaunay(pts)
            edges = tri.edges
            if len(edges) == 0:
                return np.zeros_like(pts)
            
            # Compute edge lengths
            p1 = pts[edges[:, 0]]
            p2 = pts[edges[:, 1]]
            diffs = p1 - p2
            lengths_sq = np.sum(diffs**2, axis=1)
            lengths = np.sqrt(lengths_sq)
            
            # Variance of lengths
            mean_len = np.mean(lengths)
            # We want to minimize sum((l_i - mean)^2)
            # Gradient w.r.t pts:
            # d/dx sum (l_i - mean)^2 = 2 * sum (l_i - mean) * dl_i/dx
            # dl_i/dx = (x_i - x_j) / l_i  (for edge i connecting i, j)
            
            # Efficiently accumulate gradients
            grads = np.zeros_like(pts)
            # (l_i - mean) * (1/l_i) = (l_i - mean) / l_i = 1 - mean/l_i
            # Let w_i = 2 * (l_i - mean) / l_i
            w = 2.0 * (lengths - mean_len) / np.maximum(lengths, 1e-9)
            
            # For each edge (u, v), grad_u += w_i * (p_u - p_v), grad_v -= w_i * (p_u - p_v)
            # Note: diffs = p1 - p2 = p_u - p_v
            # So we add w_i * diffs to p_u and subtract w_i * diffs from p_v
            
            # np.add.at is slow, but N=300 is small enough.
            # We can use scatter-add logic or just loop if necessary, but let's try vectorized add.at
            # w is shape (E,), diffs is shape (E, 2)
            # We need to scale each diff row by its corresponding w
            
            scaled_diffs = diffs * w[:, np.newaxis]
            
            # Add to u indices
            np.add.at(grads, edges[:, 0], scaled_diffs)
            # Subtract from v indices
            np.add.at(grads, edges[:, 1], -scaled_diffs)
            
            return grads * weight
        except Exception:
            return np.zeros_like(pts)

    # Optimization parameters
    alpha = 0.02
    cooling = 0.99995  # Slower cooling
    k_soft_start = 3.0
    k_soft_end = 150.0
    max_iter = 8000
    iter_count = 0
    stall_count = 0
    stall_threshold = 150
    
    # Adaptive k_soft
    k_soft = k_soft_start
    
    # FISTA parameters
    t_k = 1.0
    y = points.copy()
    spectral_weight = 0.05 # Start with small weight, increase if needed
    
    # Phase 1: Main optimization with FISTA and Spectral Regularization
    while time.time() - start_time < time_limit * 0.65 and iter_count < max_iter:
        # Compute pairwise distances efficiently
        diff = y[:, np.newaxis, :] - y[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        rad = np.linalg.norm(y, axis=1)
        bound_dists = 1.0 - rad
        
        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = y.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Compute Soft-Min Weights
        weights, _ = _soft_min_weights(all_vals, k_soft)
        
        # Compute Gradients for Packing Constraint
        grads = np.zeros_like(y)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (y / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Add Spectral Regularization Gradient
        spec_grad = _spectral_regularization_grad(y, spectral_weight)
        grads += spec_grad
        
        # Normalize gradient to prevent large steps
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        # FISTA Update
        # x_new = y - alpha * grad(y)
        # Project to feasible set
        x_new = y - alpha * grads
        rad_new = np.linalg.norm(x_new, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            x_new[mask] /= rad_new[mask, np.newaxis]
        
        # Momentum update
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
        y = x_new + ((t_k - 1.0) / t_next) * (x_new - y)
        
        # Project y back to feasible set as well to keep it valid
        rad_y = np.linalg.norm(y, axis=1)
        mask_y = rad_y > 1.0
        if np.any(mask_y):
            y[mask_y] /= rad_y[mask_y, np.newaxis]
        
        points = x_new
        t_k = t_next
        
        # Cool down
        alpha *= cooling
        
        # Increase k_soft over time with smoother schedule
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 1.5)
        
        # Increase spectral weight slightly as we converge to enforce structure
        spectral_weight = 0.05 + 0.05 * progress
        
        # Random perturbation to escape local minima
        if stall_count > stall_threshold:
            noise_scale = alpha * 2.0
            y += np.random.normal(0, noise_scale, size=y.shape)
            rad_new = np.linalg.norm(y, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                y[mask] /= rad_new[mask, np.newaxis]
            points = y.copy()
            t_k = 1.0
            stall_count = 0
            alpha = max(alpha, 0.005)
        
        iter_count += 1

    # Phase 2: High-precision polish
    points = best_points.copy()
    polish_alpha = 0.0005
    polish_k = 300.0
    polish_time_limit = time_limit * 0.97
    
    while time.time() - start_time < polish_time_limit:
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
        
        # Very sharp soft-min for polish
        weights, _ = _soft_min_weights(all_vals, polish_k)
        
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
        
        # Add small spectral regularization in polish too? Maybe not, focus on min distance.
        # Let's stick to pure min distance gradient for polish to maximize r.
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += polish_alpha * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
        
        polish_alpha *= 0.9995
        polish_k *= 1.002

    return best_points
