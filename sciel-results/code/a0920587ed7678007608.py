import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Initialize with a grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Small random jitter to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)

    # Log-Sum-Exp parameters
    # Start with low beta (smooth) and increase to high beta (sharp/min-finding)
    beta_start = 10.0
    beta_end = 500.0
    n_iter = 2000
    lr_start = 0.05
    lr_end = 0.001
    
    for it in range(n_iter):
        # Annealing beta
        t = it / (n_iter - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        # Annealing learning rate
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Compute distances
        # Pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists = dists[mask]
        
        # Wall distances
        # dist to 4 walls for each point
        # x: 0.5 - |x|, y: 0.5 - |y|
        dist_x = 0.5 - np.abs(centers[:, 0])
        dist_y = 0.5 - np.abs(centers[:, 1])
        wall_dists = np.concatenate([dist_x, dist_y])
        
        # Combine all distances
        all_dists = np.concatenate([dists, wall_dists])
        
        # Calculate weights for Log-Sum-Exp
        # To avoid overflow/underflow, shift by max dist? 
        # Actually, we want weights proportional to exp(-beta * d).
        # Since d > 0, large d -> small weight.
        # We can compute log_weights = -beta * all_dists
        log_weights = -beta * all_dists
        # Normalize to prevent numerical issues (softmax)
        log_weights_max = np.max(log_weights)
        weights = np.exp(log_weights - log_weights_max)
        sum_weights = np.sum(weights)
        
        # Compute Gradient
        # grad = sum(weights * grad_d_k) / sum_weights
        grad = np.zeros_like(centers)
        
        # 1. Pairwise gradients
        # Reconstruct full distance matrix for indexing
        dists_mat = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dists_mat, np.inf)
        
        # We need to map the flat `dists` back to indices.
        # It's easier to iterate over the flat array or reconstruct the weight matrix.
        # Let's reconstruct the weight matrix for pairs.
        log_weights_mat = -beta * dists_mat
        log_weights_mat_max = np.max(log_weights_mat[~np.eye(N, dtype=bool)])
        weights_mat = np.exp(log_weights_mat - log_weights_mat_max)
        weights_mat[~np.eye(N, dtype=bool)] = weights_mat[~np.eye(N, dtype=bool)]
        weights_mat[np.eye(N, dtype=bool)] = 0.0
        
        # Gradient contribution from pairs:
        # For pair (i,j), grad_i += w_ij * (c_i - c_j) / d_ij
        # grad_j += w_ij * (c_j - c_i) / d_ij
        # We can vectorize this:
        # diff_mat = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # unit_vecs = diff_mat / dists_mat[:,:,np.newaxis]
        # weighted_diffs = weights_mat[:,:,np.newaxis] * unit_vecs
        # grad_pairs = np.sum(weighted_diffs, axis=1)
        
        diff_mat = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # Avoid division by zero
        dists_safe = np.where(dists_mat > 1e-8, dists_mat, 1.0)
        unit_vecs = diff_mat / dists_safe[:, :, np.newaxis]
        # Zero out diagonal
        unit_vecs[np.eye(N, dtype=bool)] = 0.0
        weighted_diffs = weights_mat[:, :, np.newaxis] * unit_vecs
        grad_pairs = np.sum(weighted_diffs, axis=1)
        
        # 2. Wall gradients
        # Wall distances:
        # Left (x<0): d = 0.5 + x, grad = [1, 0]
        # Right (x>0): d = 0.5 - x, grad = [-1, 0]
        # Bottom (y<0): d = 0.5 + y, grad = [0, 1]
        # Top (y>0): d = 0.5 - y, grad = [0, -1]
        
        # We have wall_dists = [dist_x for all i, dist_y for all i]
        # dist_x[i] = 0.5 - abs(centers[i,0])
        # If centers[i,0] < 0, dist = 0.5 + x, grad_x = 1
        # If centers[i,0] >= 0, dist = 0.5 - x, grad_x = -1
        
        # Reconstruct weights for walls
        # log_weights_x = -beta * dist_x
        # log_weights_y = -beta * dist_y
        # Combine into a single weight vector for the 2N wall terms?
        # The total sum_weights includes both pairs and walls.
        # So we need to normalize the wall weights by the TOTAL sum_weights.
        
        # Calculate wall weights
        log_w_x = -beta * dist_x
        log_w_y = -beta * dist_y
        
        # Shift all log weights to common scale to combine correctly?
        # The previous `sum_weights` was calculated from `all_dists`.
        # To be consistent, we should calculate the global max log weight.
        
        # Let's redo the weight calculation globally to be safe and consistent.
        # all_dists = [dists (flat), wall_dists]
        # log_w_all = -beta * all_dists
        # max_log_w = max(log_w_all)
        # w_all = exp(log_w_all - max_log_w)
        # sum_w = sum(w_all)
        
        # We already have `weights` (flat, shifted by local max of all_dists).
        # Let's extract the weights for pairs and walls from the global calculation.
        
        # Recalculate global weights properly
        log_w_all = -beta * all_dists
        max_log_w = np.max(log_w_all)
        w_all = np.exp(log_w_all - max_log_w)
        sum_w = np.sum(w_all)
        
        # Split w_all into pair_weights (flat) and wall_weights (flat)
        n_pairs = N * (N - 1) // 2
        pair_w_flat = w_all[:n_pairs]
        wall_w_flat = w_all[n_pairs:]
        
        # Pair weights matrix
        # Need to map flat index to (i, j)
        # This is slow in a loop. Let's use the matrix approach with global normalization.
        
        # Re-calculate pair weights matrix with global shift
        log_w_mat = -beta * dists_mat
        # Subtract global max_log_w
        w_mat = np.exp(log_w_mat - max_log_w)
        w_mat[np.eye(N, dtype=bool)] = 0.0
        
        # Recalculate wall weights with global shift
        w_x = np.exp(-beta * dist_x - max_log_w)
        w_y = np.exp(-beta * dist_y - max_log_w)
        
        # Gradient from pairs
        # grad_i = sum_j w_ij * (c_i - c_j) / d_ij
        # We have w_mat and unit_vecs (which is (c_i - c_j) / d_ij)
        grad_pairs = np.sum(w_mat[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Gradient from walls
        grad_walls = np.zeros_like(centers)
        # x-component
        # If x < 0, grad_x = +1 * w_x
        # If x >= 0, grad_x = -1 * w_x
        sign_x = np.where(centers[:, 0] < 0, 1.0, -1.0)
        grad_walls[:, 0] += sign_x * w_x
        
        # y-component
        sign_y = np.where(centers[:, 1] < 0, 1.0, -1.0)
        grad_walls[:, 1] += sign_y * w_y
        
        # Total Gradient
        grad = grad_pairs + grad_walls
        grad /= sum_w # Normalize by total weight sum
        
        # Update
        centers += lr * grad
        
        # Clamp to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optional: small noise to escape local minima
        if it % 50 == 0 and it > 0:
            centers += np.random.normal(0, 0.0005, centers.shape)
            centers = np.clip(centers, -0.5, 0.5)

    return centers
