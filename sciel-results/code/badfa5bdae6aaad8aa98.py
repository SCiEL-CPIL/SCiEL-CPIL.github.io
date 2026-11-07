# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix for ant colony orienteering.
    
    Strategy: Use 'Prize Density' heuristic.
    Instead of just looking at the immediate prize of the destination node,
    we estimate the total prize accessible from that node within a short radius.
    This guides ants to clusters of high-value nodes.
    
    Heuristic for edge (i, j):
    H[i, j] = (NodeValue[j]) / (distance[i, j] + eps) * exp(-alpha * distance[j, 0])
    
    Where NodeValue[j] = sum(prizes[k] * exp(-beta * distance[j, k]) for all k)
    """
    n = len(prizes)
    eps = 1e-9
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Parameters
    beta = 5.0   # Decay rate for prize density (higher = more local)
    alpha = 1.0  # Penalty for distance from start (higher = safer)
    
    # 1. Compute NodeValue for each node j
    # NodeValue[j] represents the "wealth" of the neighborhood around j
    # We use a Gaussian kernel for smooth decay
    # distance_matrix is (n, n). 
    # We want for each j, sum over k of prizes[k] * exp(-beta * d[j, k])
    
    # exp_kernel[j, k] = exp(-beta * d[j, k])
    exp_kernel = np.exp(-beta * distance_matrix)
    
    # NodeValue[j] = sum_k (prizes[k] * exp_kernel[j, k])
    # This is a matrix-vector product: exp_kernel @ prizes
    node_values = exp_kernel @ prizes
    
    # 2. Compute the base heuristic for each edge (i, j)
    # We want high values for edges that lead to high-value nodes with low cost.
    
    # Cost component: 1 / (d[i, j] + eps)
    # Using a soft inverse to avoid huge spikes for very small distances
    # d[i, j] is 0 when i == j, but ants don't move to visited nodes, and 
    # typically i != j in the transition. However, diagonal entries should be low.
    dist_inv = 1.0 / (distance_matrix + eps)
    
    # Start penalty: exp(-alpha * d[j, 0])
    # Nodes far from start are riskier to visit if we need to return.
    d_to_start = distance_matrix[:, 0]
    start_penalty = np.exp(-alpha * d_to_start)
    
    # Combine: H[i, j] = node_values[j] * dist_inv[i, j] * start_penalty[j]
    # node_values[j] is shape (n,), broadcast to (n, n) along rows?
    # No, node_values[j] depends on column j.
    # So we need node_values[None, :] to broadcast across rows.
    
    heuristic = node_values[None, :] * dist_inv * start_penalty[None, :]
    
    # 3. Clean up values
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    # 4. Normalize
    # The magnitude should be reasonable relative to pheromone (starts at 1).
    # Let's normalize so the median is 1.
    median_val = np.median(heuristic)
    if median_val > eps:
        heuristic = heuristic / median_val
    else:
        heuristic = np.ones_like(heuristic)
        
    return heuristic
# EVOLVE-BLOCK-END
