"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ACO orienteering.
    
    The heuristic encourages transitions (i -> j) that:
    1. Lead to nodes with high prizes.
    2. Keep the total path length (current length + d[i,j] + d[j,0]) well within max_len.
    3. Are geometrically efficient (short d[i,j]).
    
    Key insight: In orienteering with a length constraint, visiting nodes far from
    the start (node 0) is risky because it consumes budget. A good heuristic should
    weight prizes by the "affordability" of visiting that node.
    
    We define the heuristic for edge (i, j) as:
    h[i, j] = prize[j] * exp(-lambda * d[j, 0]) / (d[i, j] + epsilon)^alpha
    
    Where:
    - prize[j] is the prize at node j.
    - d[j, 0] is the distance from node j to the start.
    - d[i, j] is the distance from i to j.
    - lambda controls the strength of the preference for nodes close to the start.
    - alpha controls how strongly we penalize long steps.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    
    eps = 1e-9
    
    # d_j0: distance from each node j to node 0 (shape: (n,))
    d_j0 = distance_matrix[:, 0]
    
    # Scale parameter for exponential decay
    # We want nodes within max_len/2 to be strongly preferred
    scale = max_len / 2.0
    
    # Affinity for node j: prize[j] * exp(-d_j0[j] / scale)
    # This downweights nodes far from start.
    node_affinity = prizes * np.exp(-d_j0 / scale)
    
    # Edge heuristic: node_affinity[j] / (d[i, j] + eps)^alpha
    # Using alpha=1.5 to more strongly penalize long steps
    alpha = 1.5
    
    # Reshape node_affinity for broadcasting: (1, n)
    node_affinity_row = node_affinity[None, :]
    
    # Distance matrix is (n, n)
    # Add epsilon to avoid division by zero
    d_plus_eps = distance_matrix + eps
    
    # h[i, j] = node_affinity[j] / d[i, j]^alpha
    h = node_affinity_row / (d_plus_eps ** alpha)
    
    # Ensure all values are positive
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
