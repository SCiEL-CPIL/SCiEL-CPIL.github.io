import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialization: Jittered Grid
    # 5x5x4 grid
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    # Create grid points
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    points = np.array(points)
    
    # Add small noise to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    
    # Clip to cube
    points = np.clip(points, -0.5, 0.5)
    
    # Precompute pairs
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    
    # Wall indices for each point
    # 6 walls per point
    # We will handle wall gradients separately or include in the soft-min set
    
    # Optimization parameters
    beta = 10.0
    step_size = 0.01
    best_radius = 0.0
    best_points = points.copy()
    
    # Helper to compute radius
    def compute_radius(pts):
        # Pair distances
        d_ij = np.linalg.norm(pts[i_idx] - pts[j_idx], axis=1)
        min_pair_dist = np.min(d_ij)
        
        # Wall distances
        # dist to -0.5: pts + 0.5
        # dist to 0.5: 0.5 - pts
        wall_dists = np.min(np.minimum(pts + 0.5, 0.5 - pts), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    # Initial evaluation
    current_radius = compute_radius(points)
    if current_radius > best_radius:
        best_radius = current_radius
        best_points = points.copy()
    
    while time.time() < deadline:
        # Compute distances
        d_vecs = points[i_idx] - points[j_idx]
        d_ij = np.linalg.norm(d_vecs, axis=1)
        
        # Avoid division by zero
        d_ij_safe = np.where(d_ij < 1e-10, 1e-10, d_ij)
        
        # Pair gradients (unit vectors)
        # grad_i s_ij = 0.5 * (c_i - c_j) / d_ij
        # grad_j s_ij = 0.5 * (c_j - c_i) / d_ij
        unit_vecs = d_vecs / d_ij_safe[:, np.newaxis]
        
        # Wall distances and gradients
        # For each point, 6 distances
        # w_k = min(0.5 - c_k, c_k + 0.5) for each dim
        # Actually, the set of values for soft-min includes ALL 6 wall distances per point?
        # Or just the min per point?
        # The objective is min( min_pair_d/2, min_wall_d ).
        # min_wall_d = min_k ( min(0.5-c_k, c_k+0.5) ).
        # This is equivalent to min over all 6N wall distances.
        
        # Let's form the full set of values S
        # S_pairs = d_ij / 2
        # S_walls = all 6N wall distances
        
        S_pairs = d_ij / 2.0
        
        # Wall distances
        # Shape (N, 6)
        # dims 0,1,2 -> pos, neg
        # dist to +0.5: 0.5 - c
        # dist to -0.5: c + 0.5
        wall_dists_2d = np.zeros((N, 6))
        wall_dists_2d[:, 0] = 0.5 - points[:, 0]
        wall_dists_2d[:, 1] = points[:, 0] + 0.5
        wall_dists_2d[:, 2] = 0.5 - points[:, 1]
        wall_dists_2d[:, 3] = points[:, 1] + 0.5
        wall_dists_2d[:, 4] = 0.5 - points[:, 2]
        wall_dists_2d[:, 5] = points[:, 2] + 0.5
        
        S_walls = wall_dists_2d.flatten()
        
        # Combine S
        S = np.concatenate([S_pairs, S_walls])
        
        # Weights
        # E = -beta * S
        # To prevent overflow, subtract max E (which is -beta * min S)
        # Since S >= 0 (if valid), min S is small, max E is large positive?
        # If S is small, E is close to 0.
        # If S is large, E is large negative.
        # Max E corresponds to min S.
        E = -beta * S
        E_max = np.max(E)
        W = np.exp(E - E_max)
        
        # Normalize weights
        W_sum = np.sum(W)
        W_norm = W / W_sum
        
        # Gradients
        # grad_L = sum (W_norm_k * grad s_k)
        
        # Pair gradients
        # For each pair k, grad s_k = 0.5 * unit_vec
        # Contribution to point i: 0.5 * unit_vec * W_pair_k
        # Contribution to point j: -0.5 * unit_vec * W_pair_k
        
        W_pairs = W_norm[:len(S_pairs)]
        
        grad_points = np.zeros((N, 3))
        
        # Accumulate pair gradients
        # Using bincount or scatter
        # i_idx, j_idx
        # contrib_i = 0.5 * unit_vecs * W_pairs[:, np.newaxis]
        # contrib_j = -0.5 * unit_vecs * W_pairs[:, np.newaxis]
        
        contrib_i = 0.5 * unit_vecs * W_pairs[:, np.newaxis]
        contrib_j = -0.5 * unit_vecs * W_pairs[:, np.newaxis]
        
        # Scatter add
        np.add.at(grad_points, i_idx, contrib_i)
        np.add.at(grad_points, j_idx, contrib_j)
        
        # Wall gradients
        # For each wall distance w_ik, grad s_k with respect to c_i
        # If w_ik = 0.5 - c_ix, grad = -1 in x
        # If w_ik = c_ix + 0.5, grad = 1 in x
        # etc.
        
        W_walls = W_norm[len(S_pairs):]
        
        # Reshape W_walls to (N, 6)
        W_walls_2d = W_walls.reshape(N, 6)
        
        # Gradient vectors for walls
        # Order: +x, -x, +y, -y, +z, -z
        # w0 = 0.5 - x -> grad (-1, 0, 0)
        # w1 = x + 0.5 -> grad (1, 0, 0)
        # w2 = 0.5 - y -> grad (0, -1, 0)
        # w3 = y + 0.5 -> grad (0, 1, 0)
        # w4 = 0.5 - z -> grad (0, 0, -1)
        # w5 = z + 0.5 -> grad (0, 0, 1)
        
        # grad_wall = W_walls_2d * grad_vecs
        # grad_vecs shape (6, 3)
        grad_vecs = np.array([
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1]
        ])
        
        # Weighted sum of grad_vecs for each point
        # grad_wall_i = sum_k W_walls_ik * grad_vecs_k
        # This is (N, 6) @ (6, 3) = (N, 3)
        grad_wall_points = W_walls_2d @ grad_vecs
        
        grad_points += grad_wall_points
        
        # Update
        points += step_size * grad_points
        
        # Clip to cube
        points = np.clip(points, -0.5, 0.5)
        
        # Evaluate
        current_radius = compute_radius(points)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Adaptive parameters
        # Increase beta to sharpen
        beta *= 1.05
        # Decrease step size
        step_size *= 0.99
        
        # If step_size is too small, restart with new random noise?
        # For now, just continue.
        
    return best_points
