import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Sample points on the sphere for evaluating the covering radius
    # Use a dense Fibonacci spiral for samples too
    M = 5000
    i_s = np.arange(M) + 0.5
    z_s = 1.0 - 2.0 * i_s / M
    phi_s = np.pi * (1.0 + np.sqrt(5.0)) * i_s
    r_s = np.sqrt(1.0 - z_s * z_s)
    S = np.column_stack([r_s * np.cos(phi_s), r_s * np.sin(phi_s), z_s])
    
    # Precompute S for dot products
    # S shape (M, 3), P shape (N, 3)
    
    beta = 1.0
    learning_rate = 0.01
    momentum = 0.9
    v = np.zeros_like(P)
    
    best_P = P.copy()
    best_score = np.inf
    
    # Helper to compute covering radius (max angular distance)
    def covering_radius(pts):
        # pts (N, 3), S (M, 3)
        # Dot product S @ pts.T -> (M, N)
        dots = S @ pts.T
        # Clip to avoid numerical issues in arccos
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        # Min distance for each sample point
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    # Initial score
    current_score = covering_radius(P)
    best_score = current_score
    
    num_iterations = 0
    while time.time() < deadline:
        # Compute dot products
        dots = S @ P.T  # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # (M, N)
        
        # Soft-max weights
        # To avoid overflow, subtract max dist for each row
        max_dists = np.max(dists, axis=1, keepdims=True)
        exp_dists = np.exp(beta * (dists - max_dists))
        Z = np.sum(exp_dists, axis=1, keepdims=True)  # (M, 1)
        weights = exp_dists / Z  # (M, N)
        
        # Gradient of soft-max distance
        # d(d_ij)/dP_j = -1/sqrt(1 - x_ij^2) * S_i
        # Note: sqrt(1 - x^2) is sin(arccos(x))
        sin_dists = np.sqrt(np.clip(1.0 - dots * dots, 1e-12, 1.0))
        grad_dists = -S[:, np.newaxis, :] / sin_dists[:, :, np.newaxis]  # (M, N, 3)
        
        # Weighted sum of gradients for each P_j
        # grad_P[j] = sum_i weights[i, j] * grad_dists[i, j]
        # Using einsum: 'ij,ijk->jk'
        grad_P = np.einsum('ij,ijk->jk', weights, grad_dists)
        
        # Update momentum
        v = momentum * v - learning_rate * grad_P
        P = P + v
        
        # Normalize P
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        P = P / norms
        
        # Anneal beta
        beta *= 1.01
        
        # Adjust learning rate
        learning_rate *= 0.99
        
        # Check score periodically
        num_iterations += 1
        if num_iterations % 50 == 0:
            current_score = covering_radius(P)
            if current_score < best_score:
                best_score = current_score
                best_P = P.copy()
        
        # Early stop if learning rate is too small
        if learning_rate < 1e-5:
            break

    # Final polish: try a few local moves or just return best_P
    # If time permits, do a simple hill climbing on the raw max
    if time.time() < deadline - 10:
        lr = 1e-3
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            dots = S @ P.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            # Find the sample point with the max min-dist
            worst_idx = np.argmax(min_dists)
            worst_dist = min_dists[worst_idx]
            
            # Find the nearest point to this sample
            nearest_p_idx = np.argmin(dists[worst_idx])
            
            # Move the nearest point towards the worst sample
            # Gradient step towards reducing the distance
            s_worst = S[worst_idx]
            p_near = P[nearest_p_idx]
            
            # Direction on sphere
            # Move p_near towards s_worst
            # Vector addition and normalization
            new_p = p_near + lr * s_worst
            new_p = new_p / np.linalg.norm(new_p)
            
            # Check if it improves
            # Quick check: just update and see
            P[nearest_p_idx] = new_p
            
            # If we want to be safe, we could check the score, but it's expensive.
            # Assuming local improvement is likely.
            
            lr *= 0.99

    # Final check
    final_score = covering_radius(P)
    if final_score < best_score:
        best_P = P.copy()
        
    return best_P
