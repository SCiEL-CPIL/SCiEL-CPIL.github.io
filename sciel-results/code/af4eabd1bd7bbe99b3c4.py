"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    Structural change: use local constraint-aware value density (prize/cost^2)
    to directly align with the OPTW objective of maximizing prize under length constraint.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Distance to origin for each node
    d_to_origin = distance_matrix[0, :]
    
    # For each node j, compute the "cost" of visiting it:
    # cost_j = d[i,j] + d[j,0] for a transition i->j
    # We want to favor high prize / low cost combinations
    
    # Value density: prize_j / (d[i,j] + d[j,0])^2
    # This rewards high-value nodes that are cheap to reach and return from
    
    # Build cost matrix: cost[i,j] = d[i,j] + d[j,0]
    cost_matrix = distance_matrix + d_to_origin[None, :]
    
    # Prize term: use prize directly (no sqrt) to emphasize high-value targets
    prize_term = prizes[None, :] + 1e-9
    
    # Value density: prize / cost^2
    # This is the core structural change: directly models prize-per-unit-cost-squared
    value_density = prize_term / (cost_matrix + 1e-9) ** 2
    
    # Add a small distance penalty to break ties and prefer shorter edges
    # when value densities are similar
    dist_penalty = 1.0 / (distance_matrix + 1e-9) ** 0.5
    
    # Combine: value_density * dist_penalty
    heuristic = value_density * dist_penalty
    
    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
