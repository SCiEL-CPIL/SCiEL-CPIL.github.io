import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses gradient ascent with a LogSumExp approximation of the minimum distance to focus on bottleneck pairs.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_score = -1.0
    best_points = None

    n_restarts = 50
    max_steps = 2000
    # Lambda controls the sharpness of the min approximation.
    # Higher lambda focuses more on the absolute minimum.
    # We can anneal lambda or use a fixed high value.
    lambda_min = 10.0 
    
    # Initial learning rate
    lr_init = 0.05
    lr_decay = 0.999

    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Deterministic random init
        np.random.seed(1000 + r)
        
        # Generate random points on sphere
        vecs = np.random.randn(n, d)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        points = vecs / norms
        
        lr = lr_init
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Compute pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Mask diagonal
            mask = ~np.eye(n, dtype=bool)
            dists_masked = np.where(mask, dists, np.inf)
            
            # LogSumExp approximation of min distance
            # We want to maximize min(d_ij).
            # Gradient of -log(sum(exp(-lambda * d_ij))) w.r.t p_i
            # = (sum_{j != i} exp(-lambda * d_ij) * grad_{p_i} d_ij) / sum_{j != i} exp(-lambda * d_ij)
            
            # Weights w_ij = exp(-lambda * d_ij)
            # Note: dists has 0 on diagonal. exp(0)=1. We must exclude diagonal.
            weights = np.exp(-lambda_min * dists)
            np.fill_diagonal(weights, 0.0)
            
            # grad_{p_i} d_ij = (p_i - p_j) / d_ij
            # We need sum_j w_ij * (p_i - p_j) / d_ij
            
            # Safe division
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            np.fill_diagonal(dists_safe, 1.0)
            
            # Unit vectors u_ij = (p_i - p_j) / d_ij
            unit_diffs = diff / dists_safe[:, :, np.newaxis]
            
            # Weighted sum
            # grad_i = sum_j w_ij * u_ij
            grads = np.sum(weights[:, :, np.newaxis] * unit_diffs, axis=1)
            
            # Normalize gradients to prevent large steps if weights are large?
            # The weights sum to at most n-1.
            # It's a weighted average of unit vectors, so magnitude <= 1.
            
            # Project onto tangent plane
            dots = np.sum(grads * points, axis=1, keepdims=True)
            grads_tangent = grads - dots * points
            
            # Update
            points = points + lr * grads_tangent
            
            # Normalize to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            # Decay learning rate
            lr *= lr_decay

        # Calculate score
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max > 0:
            score = (d_min / d_max) ** 2
        else:
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_points = points.copy()

    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        
    return best_points
