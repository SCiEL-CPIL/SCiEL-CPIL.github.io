"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix for ACO TSP.
    
    Uses a combination of inverse distance and degree-awareness to guide ants.
    The base heuristic is 1/d^2, which strongly favors short edges. We then
    apply a mild penalty based on the number of very-short edges incident to
    each node, to encourage a more balanced tour structure and avoid premature
    convergence to star-like patterns.
    """
    start_time = time.time()
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Safety check for deadline
    if time.time() - start_time > 290:
        return 1.0 / (d + 1e-9)
    
    # Replace diagonal (self-loops) with a large value to make their heuristic ~0
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Base heuristic: inverse squared distance
    # Strong preference for short edges, but not as extreme as d^5
    eta = 1.0 / (d_safe ** 2 + 1e-18)
    
    # Degree-awareness penalty:
    # Count how many of the 5 nearest neighbors each node has
    # Nodes with many very-close neighbors get a mild penalty to prevent
    # ants from over-using them, which can lead to poor tour structures
    
    # For each node, find its 5 nearest neighbors
    # This is O(n^2 log n) which is fine for n=50
    k = 5
    # Sort distances for each node (excluding self)
    # d_safe has 1e10 on diagonal, so it'll sort last
    sorted_d = np.sort(d_safe, axis=1)
    
    # The 5th nearest neighbor distance for each node
    # If a node has many very close neighbors, its 5th nearest is small
    # We want to penalize nodes that are "too popular"
    
    # Compute a popularity score: how close are the 5 nearest neighbors?
    # Lower 5th-nearest distance = more clustered/popular node
    fifth_nearest = sorted_d[:, k]  # 0-indexed, so index 4 is 5th
    
    # Normalize: nodes with smaller fifth_nearest are more "popular"
    # We want to penalize very popular nodes slightly
    # Use a soft penalty: divide by (1 + popularity_weight * normalized_popularity)
    
    # Normalize fifth_nearest to [0, 1]
    min_fn = np.min(fifth_nearest)
    max_fn = np.max(fifth_nearest)
    if max_fn > min_fn:
        norm_fn = (fifth_nearest - min_fn) / (max_fn - min_fn)
    else:
        norm_fn = np.zeros_like(fifth_nearest)
    
    # Popularity = 1 - norm_fn (higher for more clustered nodes)
    popularity = 1.0 - norm_fn
    
    # Mild penalty: divide heuristic by (1 + alpha * popularity)
    # alpha = 0.3 is a gentle adjustment
    alpha = 0.3
    penalty = 1.0 / (1.0 + alpha * popularity)  # shape (n,)
    
    # Apply penalty: eta[i,j] *= penalty[i] * penalty[j]
    # This penalizes edges incident to very popular nodes
    eta = eta * penalty[:, np.newaxis] * penalty[np.newaxis, :]
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
