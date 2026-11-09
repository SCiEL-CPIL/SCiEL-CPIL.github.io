# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit, leave margin

    def score(points):
        if np.any(np.isnan(points)):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def gradient_ascent(points, lr, soft_beta):
        """
        Gradient ascent on the smooth surrogate:
        maximize soft_min(dists) / soft_max(dists)
        using log-sum-exp for stability.
        """
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return points, 0.0

        # Soft min and max using log-sum-exp
        # soft_min = -log(sum(exp(-beta * d))) / beta
        # soft_max = log(sum(exp(beta * d))) / beta
        
        # For numerical stability
        neg_d = -dists
        max_neg_d = np.max(neg_d)
        log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
        soft_min = -log_sum_exp_neg / soft_beta
        
        max_d = np.max(dists)
        log_sum_exp_pos = max_d + np.log(np.sum(np.exp(dists - max_d)))
        soft_max = log_sum_exp_pos / soft_beta
        
        if soft_max == 0:
            return points, 0.0
        
        # Objective: soft_min / soft_max
        # Gradient of soft_min w.r.t. point i
        # d(soft_min)/d(point_i) = sum_j w_j * (point_i - point_j) / ||point_i - point_j||
        # where w_j = exp(-beta * d_ij) / sum_k exp(-beta * d_ik) for j != i
        
        # Similarly for soft_max
        # d(soft_max)/d(point_i) = sum_j v_j * (point_i - point_j) / ||point_i - point_j||
        # where v_j = exp(beta * d_ij) / sum_k exp(beta * d_ik) for j != i
        
        grad = np.zeros_like(points)
        
        # Compute weights for soft_min
        # For each pair (i, j), weight_ij = exp(-beta * d_ij)
        # We need to compute gradient contributions
        
        # Efficient computation:
        # For each point i, grad_i = (1/soft_max) * d(soft_min)/d(point_i) - (soft_min/soft_max^2) * d(soft_max)/d(point_i)
        
        # Let's compute d(soft_min)/d(point_i) and d(soft_max)/d(point_i)
        
        # d(soft_min)/d(point_i) = -1/beta * d(log_sum_exp_neg)/d(point_i)
        # log_sum_exp_neg = log(sum_j exp(-beta * d_ij))
        # d(log_sum_exp_neg)/d(point_i) = sum_j exp(-beta * d_ij) * (-beta) * d(d_ij)/d(point_i) / sum_k exp(-beta * d_ik)
        # d(d_ij)/d(point_i) = (point_i - point_j) / d_ij
        
        # So d(soft_min)/d(point_i) = sum_j exp(-beta * d_ij) * (point_i - point_j) / d_ij / sum_k exp(-beta * d_ik)
        
        # Similarly d(soft_max)/d(point_i) = sum_j exp(beta * d_ij) * (point_i - point_j) / d_ij / sum_k exp(beta * d_ik)
        
        # Compute pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists_matrix, np.inf)
        
        # Weights for soft_min: w_ij = exp(-beta * d_ij)
        w_min = np.exp(-soft_beta * dists_matrix)
        np.fill_diagonal(w_min, 0)
        sum_w_min = np.sum(w_min, axis=1, keepdims=True)
        
        # Weights for soft_max: v_ij = exp(beta * d_ij)
        v_max = np.exp(soft_beta * dists_matrix)
        np.fill_diagonal(v_max, 0)
        sum_v_max = np.sum(v_max, axis=1, keepdims=True)
        
        # Avoid division by zero
        sum_w_min = np.maximum(sum_w_min, 1e-300)
        sum_v_max = np.maximum(sum_v_max, 1e-300)
        
        # d(soft_min)/d(point_i) = sum_j w_ij * (point_i - point_j) / d_ij / sum_k w_ik
        # = sum_j (w_ij / d_ij) * (point_i - point_j) / sum_k w_ik
        
        # Compute (w_ij / d_ij) for each pair
        w_min_over_d = w_min / dists_matrix
        np.fill_diagonal(w_min_over_d, 0)
        
        # Gradient contribution from soft_min
        grad_min = np.zeros_like(points)
        for i in range(n):
            grad_min[i] = np.sum(w_min_over_d[i, :] * diff[i, :], axis=0) / sum_w_min[i, 0]
        
        # d(soft_max)/d(point_i) = sum_j v_ij * (point_i - point_j) / d_ij / sum_k v_ik
        v_max_over_d = v_max / dists_matrix
        np.fill_diagonal(v_max_over_d, 0)
        
        grad_max = np.zeros_like(points)
        for i in range(n):
            grad_max[i] = np.sum(v_max_over_d[i, :] * diff[i, :], axis=0) / sum_v_max[i, 0]
        
        # Total gradient of soft_min / soft_max
        # d(soft_min/soft_max)/d(point_i) = (soft_max * grad_min_i - soft_min * grad_max_i) / soft_max^2
        grad = (soft_max * grad_min - soft_min * grad_max) / (soft_max ** 2)
        
        # Update
        new_points = points + lr * grad
        
        # Compute score
        s = score(new_points)
        
        return new_points, s

    # Initialize points on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = score(best_points)
    
    # Multiple restarts with different initializations
    np.random.seed(123)
    
    n_restarts = 5
    restarts = []
    
    # Circle initialization
    restarts.append(best_points.copy())
    
    # Random initializations
    for _ in range(n_restarts - 1):
        pts = np.random.randn(n, d) * 2
        restarts.append(pts)
    
    # Also try regular polygon with some perturbation
    angles_pert = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.random.randn(n) * 0.1
    pts_pert = np.column_stack([np.cos(angles_pert), np.sin(angles_pert)])
    restarts.append(pts_pert)
    
    for restart_idx, init_points in enumerate(restarts):
        if time.time() > deadline - 30:
            break
        
        points = init_points.copy()
        current_score = score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Gradient ascent with annealing
        n_iterations = 5000
        initial_lr = 0.1
        final_lr = 0.001
        soft_beta = 10.0
        
        for it in range(n_iterations):
            if time.time() > deadline - 30:
                break
            
            lr = initial_lr * (final_lr / initial_lr) ** (it / n_iterations)
            
            # Increase soft_beta over time for sharper min/max
            beta_schedule = 10.0 + 50.0 * (it / n_iterations)
            
            new_points, new_score = gradient_ascent(points, lr, beta_schedule)
            
            if new_score > current_score:
                points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # If score decreased, revert and reduce learning rate
                lr *= 0.9
        
        # Local refinement with smaller steps
        if time.time() > deadline - 20:
            break
        
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            new_points, new_score = gradient_ascent(points, 0.001, 50.0)
            if new_score > current_score:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
    
    # Final check
    if time.time() > deadline - 5:
        return best_points
    
    return best_points


# EVOLVE-BLOCK-END
