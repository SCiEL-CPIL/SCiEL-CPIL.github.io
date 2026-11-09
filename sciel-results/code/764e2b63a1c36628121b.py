"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix for ACO orienteering.
    
    Uses a greedy ratio heuristic: attractiveness of moving to node j from i
    is proportional to prizes[j] / (d[i,j] + d[j,0])^alpha, where alpha
    controls distance sensitivity. This favors high-prize nodes that are
    'cheap' in terms of the total path cost (go there and return to origin).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Total cost of visiting node j from i: d[i,j] + d[j,0]
    # This represents the additional length if we go i->j and then j->0
    # (the return-to-origin cost is incurred regardless of when we visit j)
    total_cost = distance_matrix + distance_matrix[:, 0:1]
    
    # Ensure no zeros cause division issues
    total_cost_safe = np.maximum(total_cost, 1e-9)
    
    # Greedy ratio: prize density per unit of additional path cost
    # Use alpha=1.0 as a balanced choice; could tune but 1.0 is natural
    alpha = 1.0
    
    # heuristic[i,j] = prizes[j] / (total_cost[i,j])^alpha
    # This is node-centric in the prize, edge-aware in the cost
    heuristic = prizes[None, :] / (total_cost_safe ** alpha)
    
    # Ensure all entries are positive (non-positive becomes 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
