"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for Ant Colony Optimization.
    
    Strategy: Combinatorial Reachability Heuristic
    Instead of using a simple pointwise ratio of prize to distance, we estimate 
    the 'future value' of visiting a node j. This value represents the maximum 
    total prize an ant could potentially collect if it were at node j, given 
    the remaining length budget.
    
    We approximate this by solving a simplified knapsack problem for each node j:
    1. Determine the remaining budget after moving to j and returning to 0.
       Since the ant doesn't know its current position 'i' in a static heuristic 
       that depends only on 'j' (or if we make it depend on 'i' via d[i,j]), 
       we must be careful.
       
       However, the standard ACO heuristic eta[i,j] can depend on i.
       The constraint is: len_so_far + d[i,j] + d[j,0] <= max_len.
       If we assume the ant is at i, the cost to go to j and return is d[i,j] + d[j,0].
       But the ant doesn't return to 0 immediately; it continues to other nodes.
       
       A better approximation for "value of being at j":
       Let L be the total path length so far (unknown statically).
       However, we can approximate the "best case" remaining capacity.
       
       Alternative Interpretation:
       The heuristic should guide the ant to nodes that are part of high-value clusters.
       For a node j, its attractiveness should be proportional to the prize of j 
       plus the prizes of other nodes k that are "close" to j and can be visited 
       in a sequence starting from j.
       
       Let's use a 1-step lookahead greedy approximation:
       Value(j) = prize[j] + sum of prizes of top-k nodes reachable from j within budget.
       
       Since max_len is small (3.0) and N=50, we can afford some computation.
       
       For each pair (i, j):
       The move i->j consumes d[i,j] in the path.
       The return cost d[j,0] is a "tax" on the total length.
       
       Let's define the heuristic h[i,j] based on the potential prize gain from j.
       We approximate the maximum prize collectable if we start at j with a budget 
       of max_len - d[i,j] - d[j,0] (assuming this is the remaining budget for 
       further moves before returning, which is an upper bound).
       
       Actually, a simpler and robust structural change:
       Rank nodes by prize.
       For each j, calculate a "cluster score": how many high-prize nodes are within 
       a small distance from j?
       
       Let's try: h[i,j] = prize[j] * (1 + density_of_high_prize_nodes_around_j)
       Where density is the number of nodes k with prize[k] > median_prize and d[j,k] < threshold.
       
       This encourages moving to hubs of high-value nodes.
       
       To make it depend on i (as required by the matrix shape), we can penalize 
       long moves i->j slightly, but primarily focus on the "hub" quality of j.
       
       Let's implement:
       1. Identify "high value" nodes (prize > 0.5 * max_prize).
       2. For each node j, count how many high value nodes are within distance 0.5.
          Let this count be C_j.
       3. Heuristic h[i,j] = prize[j] * (1 + C_j) / (1 + d[i,j])
       
       This is structurally different: it uses local neighborhood density of high-value 
       nodes rather than just the individual node's own prize/cost ratio.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    N = len(prizes)
    
    if N == 0:
        return np.array([])
        
    # Identify high-value nodes
    max_prize = np.max(prizes)
    if max_prize == 0:
        # All prizes are 0, return uniform heuristic
        return np.ones((N, N)) * 1e-9
        
    threshold_prize = 0.5 * max_prize
    high_value_mask = prizes > threshold_prize
    
    # Calculate "hub score" for each node j:
    # Count of high-value nodes within a certain distance from j.
    # Distance threshold: 0.5 (approximate quarter of unit square diagonal)
    dist_threshold = 0.5
    
    # Vectorized count of high-value neighbors
    # For each j, count k such that high_value_mask[k] is True and distance_matrix[j, k] < dist_threshold
    # Note: distance_matrix is symmetric usually, but we treat it as directed for safety.
    
    # Create a boolean matrix of close connections
    close_connections = distance_matrix < dist_threshold
    
    # For each j, sum the high_value_mask of its close neighbors
    # hub_score[j] = sum_k (close_connections[j, k] * high_value_mask[k])
    hub_scores = np.dot(close_connections, high_value_mask.astype(float))
    
    # Normalize hub scores to [0, 1] range for stability, or keep as count
    # Let's use (1 + hub_scores) as a multiplier
    hub_multiplier = 1.0 + hub_scores
    
    # Base attractiveness: prize[j]
    # We want h[i,j] to be high if prize[j] is high AND j is a hub.
    # And slightly lower if d[i,j] is large (to prefer local moves).
    
    prize_col = prizes[None, :] * hub_multiplier[None, :]
    
    # Distance penalty: 1 / (1 + d[i,j])
    # This ensures finite values and smooth decay.
    dist_penalty = 1.0 / (1.0 + distance_matrix)
    
    h = prize_col * dist_penalty
    
    # Ensure positive and finite
    h = np.where(np.isfinite(h), h, 1e-9)
    h = np.maximum(h, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return h
        
    return h
# EVOLVE-BLOCK-END
