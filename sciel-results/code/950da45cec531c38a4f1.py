# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix for ant colony orienteering.
    
    Strategy: Use a 'Prize Density' heuristic.
    Instead of penalizing detours based on triangle inequality, we estimate
    the 'value density' around each node. Nodes that are close to many other
    high-value nodes are considered more attractive destinations because
    visiting them allows for chaining multiple prizes efficiently.
    
    1. For each node j, calculate a density score D[j] which is the sum of
       prizes of all nodes k weighted by their proximity to j.
       D[j] = sum_k prize[k] * exp(-d[j, k]^2 / sigma^2)
       
    2. The heuristic for edge (i, j) is proportional to D[j] / d[i, j].
       This encourages moving to high-density areas via short edges.
    """
    n = len(prizes)
    eps = 1e-9
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Calculate Prize Density for each node
    # sigma controls the spatial scale of the "cluster" perception.
    # A smaller sigma makes the heuristic more local.
    # max_len is 3, typical distances in unit square are < 1.5.
    # Let's set sigma based on the median distance or a fixed fraction of max_len.
    # Using a fixed value relative to the domain size (1.0) seems robust.
    sigma = 0.5
    
    # d_sq: squared distances
    d_sq = distance_matrix ** 2
    
    # Weight matrix: exp(-d_sq / sigma^2)
    # We ignore self-attraction? Or include it? 
    # Including it means high prize nodes have high density even in isolation.
    # Excluding it forces clustering. Let's include it but the prize[0]=0 handles the start.
    # Actually, for destination attractiveness, we want to know if *other* nodes are nearby.
    # But if a node has a huge prize, it should be attractive on its own.
    # So D[j] = sum_k prize[k] * w[j,k] is good.
    
    w = np.exp(-d_sq / (sigma * sigma))
    
    # D[j] = sum over k of prize[k] * w[j, k]
    # prizes shape (n,), w shape (n, n)
    # D = w.T @ prizes ? 
    # w[j, k] is weight from j to k.
    # We want D[j] = sum_k w[j, k] * prizes[k].
    # This is w @ prizes.
    D = w @ prizes
    
    # Heuristic for edge (i, j) is D[j] / d[i, j]
    # D[j] is the attractiveness of node j.
    # We want to move to j if it's attractive and close.
    
    # Create matrix where column j is D[j]
    D_col = D[None, :] # shape (1, n)
    
    # Distance matrix d[i, j]
    # Avoid division by zero for i==j (though ants don't revisit, heuristic should be safe)
    # Also, if d[i,j] is 0, we set to eps.
    safe_dist = np.where(distance_matrix > 0, distance_matrix, 1.0)
    
    heuristic = D_col / safe_dist
    
    # Handle the case where i == j. The ant cannot stay. 
    # Usually heuristics for i==j are ignored or set to 0/low.
    # Since probability is proportional to pheromone * heuristic, and pheromone[i,i] might be non-zero,
    # we should ensure heuristic[i,i] is low or 0 to prevent self-loops if the algorithm allows it.
    # The problem statement says "moves to an unvisited node j". So i!=j is enforced by the algorithm.
    # However, to be safe and clean, we can set diagonal to 0.
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    # If D[j] is 0 (e.g., node j is far from all prizes), heuristic is 0.
    # The contract says "Heuristic entries that are not positive become 1e-9".
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    # Normalize so that the mean is around 1, similar to the seed program's scaling.
    # This helps with the pheromone initialization (starts at 1).
    mean_val = np.mean(heuristic)
    if mean_val > eps:
        heuristic = heuristic / mean_val
        
    return heuristic
# EVOLVE-BLOCK-END
