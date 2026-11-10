import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ACO solver.
    
    The heuristic is designed to favor moves to nodes that:
    1. Have high prize values.
    2. Are close to the current node.
    3. Are close to the origin (start/end node), to ensure the tour can be completed within the length constraint.
    
    Formula: eta[i, j] = ( (prizes[j] + eps) / (distance[i, j] + distance[j, 0] + eps) )^2
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    eps = 1e-9
    
    # Vectorized calculation
    # prizes is (N,), distance_matrix is (N, N)
    # We want eta[i, j] to depend on prizes[j], distance[i, j], and distance[j, 0]
    
    # distances from each node j to the origin (node 0)
    dist_to_origin = distance_matrix[:, 0]  # Shape (N,)
    
    # Effective cost of moving i -> j: distance(i, j) + distance(j, 0)
    # distance_matrix is (N, N), dist_to_origin is (N,)
    # We need to broadcast dist_to_origin to align with columns (j)
    # dist_to_origin[None, :] is (1, N)
    # distance_matrix + dist_to_origin[None, :] results in (N, N) where element [i, j] is D[i, j] + D[j, 0]
    effective_cost = distance_matrix + dist_to_origin[None, :]
    
    # Prize attractiveness: prizes[j]
    # prizes[None, :] is (1, N)
    prize_attractiveness = prizes[None, :]
    
    # Heuristic calculation
    # Using a power of 2 to increase contrast, which helps in selecting high-value nodes
    # when the path length is constrained.
    ratio = (prize_attractiveness + eps) / (effective_cost + eps)
    heuristic = ratio ** 2
    
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
