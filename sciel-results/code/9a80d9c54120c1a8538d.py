"""Heuristic for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Combines:
      1. Inverse distance (base attractor)
      2. Smooth demand-compatibility penalty (Lagrangian-style soft constraint)
      3. Local spatial coherence bonus
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    
    # 1. Base: inverse distance (larger for closer edges)
    base = 1.0 / (distance_matrix + 1e-9)
    
    # 2. Demand-compatibility: smooth penalty for edges whose combined
    #    demand is too large to fit comfortably in a single vehicle load.
    #    Use a soft penalty: penalty grows as (d_i + d_j) approaches capacity.
    #    For depot edges (demand 0), no penalty.
    demand_sum = demands[:, None] + demands[None, :]
    # Normalized load: 0 = empty, 1 = full
    load_ratio = demand_sum / capacity
    # Soft penalty: smooth function that is ~0 for low loads, grows for high
    # Using a smoothstep-like function centered around 0.7*capacity
    threshold = 0.7 * capacity
    # Smooth penalty: exp(-alpha * max(0, load_ratio - threshold/capacity)^2)
    # Actually let's use: 1 / (1 + beta * max(0, load_ratio - 0.6)^2)
    # This gives a smooth reduction for edges with high combined demand
    excess = np.maximum(0.0, load_ratio - 0.6)
    penalty = 1.0 / (1.0 + 4.0 * excess * excess)
    # Depot has demand 0, so depot edges get penalty ~ 1/(1+4*(d_j/cap)^2)
    # which is mild for typical demands (1-9 out of 50)
    
    # 3. Local spatial coherence: for each node i, compute the average distance
    #    to its k nearest neighbors. Edges to nearby nodes get a small bonus.
    #    This encourages routes that stay within local clusters.
    k = 5
    # Compute distances from each node to all others
    dists = distance_matrix.copy()
    # For each node, find k-nearest neighbors
    coherence = np.ones((n, n))
    for i in range(n):
        # Get distances from node i to all others
        d_i = dists[i].copy()
        d_i[i] = np.inf
        # k nearest neighbors
        knn_idx = np.argsort(d_i)[:k]
        knn_dist = d_i[knn_idx]
        avg_knn_dist = np.mean(knn_dist) if len(knn_dist) > 0 else 1.0
        # Bonus: edges shorter than average kNN distance get boosted
        # Use a smooth ratio
        ratio = d_i / (avg_knn_dist + 1e-9)
        # Boost for close edges, mild penalty for far ones
        coherence[i, :] = 1.0 / (1.0 + 0.5 * np.maximum(0, ratio - 1.0))
        coherence[i, i] = 1.0  # no self-edge
    
    # Combine: base * penalty * coherence
    # Normalize to keep values in a reasonable range
    heuristic = base * penalty * coherence
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Symmetrize (distance matrix is symmetric, so heuristic should be too)
    heuristic = 0.5 * (heuristic + heuristic.T)
    
    return heuristic
# EVOLVE-BLOCK-END
