"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ant colony optimization
    of the orienteering problem.
    
    Structural change: feasibility-aware heuristic that estimates the maximum
    additional prize collectible after moving to node j, incorporating the
    actual length constraint. This is a different modelling family than
    simple prize/distance ratios.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    d = distance_matrix.copy()
    
    # For each pair (i, j), estimate how much additional prize can be collected
    # after visiting j, given the remaining length budget.
    # Remaining budget after going 0->i->j->0: max_len - d[0,i] - d[i,j] - d[j,0]
    # If remaining budget > 0, we can potentially visit more nodes.
    
    heuristic = np.ones((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Cost to go 0 -> i -> j -> 0
            cost_so_far = d[0, i] + d[i, j]
            return_cost = d[j, 0]
            remaining = max_len - cost_so_far - return_cost
            
            if remaining < 0:
                # Not even feasible to go back; very low attractiveness
                heuristic[i, j] = 1e-9
                continue
            
            # Base attractiveness: prize at j
            base = prizes[j]
            
            if remaining > 1e-9:
                # Estimate additional prize from remaining budget
                # For each candidate k != i, j, 0, check if d[j,k] + d[k,0] <= remaining
                # and add prize[k] * (1 / (1 + d[j,k]))
                addl = 0.0
                for k in range(n):
                    if k == i or k == j or k == 0:
                        continue
                    trip_cost = d[j, k] + d[k, 0]
                    if trip_cost <= remaining + 1e-9:
                        # Weight by inverse distance to prefer closer high-prize nodes
                        addl += prizes[k] / (1.0 + d[j, k])
                base += addl
            
            # Scale by proximity: closer nodes are slightly more attractive
            proximity = 1.0 / (1.0 + d[i, j])
            heuristic[i, j] = base * proximity
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
