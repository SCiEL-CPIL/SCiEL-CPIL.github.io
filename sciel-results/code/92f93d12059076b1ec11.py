"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses inverse distance combined with a penalty for edges that are
    likely to cross other edges (based on metric properties).
    """
    n = distance_matrix.shape[0]
    # Base heuristic: inverse distance
    # Add small epsilon to avoid division by zero
    h = 1.0 / (distance_matrix + 1e-9)
    
    # Penalty for crossing-like edges
    # We iterate over all edges (i, j)
    # For each edge, we check if it forms a "crossing" with any other edge (k, l)
    # Condition for crossing in metric space:
    # d(i, j) + d(k, l) > d(i, k) + d(j, l)
    # d(i, j) + d(k, l) > d(i, l) + d(j, k)
    
    # To speed up, we can just count how many times an edge is "longer" 
    # than the alternative pairings for any 4-tuple.
    
    # Initialize penalty factor to 1.0
    penalty = np.ones((n, n))
    
    # Vectorized approach might be tricky for the double loop over k,l.
    # Given N=50, a simple loop is acceptable if optimized.
    # 50*50 = 2500 edges.
    # For each edge, 48*47/2 ~ 1128 other edges to check.
    # Total ~ 2.8 million checks. This is fast in Python.
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            dij = distance_matrix[i, j]
            # Find k, l such that k, l not in {i, j}
            # We can just iterate all k, l
            count_cross = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                for l in range(k + 1, n):
                    if l == i or l == j:
                        continue
                    dkl = distance_matrix[k, l]
                    # Check crossing condition
                    # Note: We only need to check if (i,j) is the "long" side
                    # i.e., dij + dkl > dik + djl AND dij + dkl > dil + djk
                    if (dij + dkl > distance_matrix[i, k] + distance_matrix[j, l]) and \
                       (dij + dkl > distance_matrix[i, l] + distance_matrix[j, k]):
                        count_cross += 1
            
            # Apply penalty based on count
            # If an edge crosses many other edges, it's less likely to be in the optimal tour
            # Scale the penalty. Let's use exp(-count * factor) or 1 / (1 + count)
            # Let's try a multiplicative factor that decreases with count
            if count_cross > 0:
                # A strong penalty
                penalty[i, j] *= (1.0 / (1.0 + count_cross * 0.5))
                penalty[j, i] *= (1.0 / (1.0 + count_cross * 0.5))
            
    # Boost nearest neighbors
    # For each node, find its 3 nearest neighbors
    # If j is in NN(i) and i is in NN(j), boost
    boost = np.ones((n, n))
    for i in range(n):
        # Get sorted distances for node i
        idx = np.argsort(distance_matrix[i])
        nn_i = idx[1:4] # Top 3, excluding self
        
        for j in nn_i:
            # Check if i is in top 3 of j
            idx_j = np.argsort(distance_matrix[j])
            nn_j = idx_j[1:4]
            if i in nn_j:
                boost[i, j] *= 2.0
                boost[j, i] *= 2.0

    # Combine
    h_final = h * penalty * boost
    
    # Ensure positive and finite
    h_final = np.where(np.isfinite(h_final) & (h_final > 0), h_final, 1e-9)
    
    return h_final
# EVOLVE-BLOCK-END
