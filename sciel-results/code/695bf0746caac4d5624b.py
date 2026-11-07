"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies a guided local search penalty to the current tour edges.
    
    Strategy:
    Instead of simulating 2-opt on candidate matrices (which is computationally 
    expensive and may not perfectly predict the evaluator's 2-opt behavior), 
    we directly modify the distance matrix to penalize edges that are part of 
    the current local optimum. The penalty is proportional to how many times 
    that edge has been used in previous local optima (edge_n_used).
    
    We use a relatively high base penalty to ensure the 2-opt step on the 
    perturbed matrix is forced to break the current tour structure significantly, 
    allowing the subsequent 2-opt on true distances to explore a new basin.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    n = len(local_opt_tour)
    
    # Create a copy of the true distance matrix
    perturbed = edge_distance.copy()
    
    # Identify edges in the current local optimal tour
    # We need to penalize these edges in the perturbed matrix so that 
    # the 2-opt run on the perturbed matrix avoids them.
    
    # Base penalty strength. 
    # The edge distances are in [0, sqrt(2)] ~ [0, 1.414].
    # A penalty of ~1.0 to 2.0 is significant.
    # We scale the penalty by the usage count to implement "guidance".
    # Edges used frequently in previous local optima get higher penalties,
    # encouraging the search to move away from previously visited regions/states.
    
    base_penalty = 1.5
    count_weight = 0.5
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Get the number of times this edge has been in a local optimum
        # edge_n_used is symmetric, so we can check [u, v]
        count = edge_n_used[u, v]
        
        # Penalty = base + weight * count
        # This ensures even unused edges in the current tour get a kick,
        # but frequently used "bad" edges get a stronger kick.
        penalty = base_penalty + count_weight * count
        
        # Apply to both directions
        perturbed[u, v] += penalty
        perturbed[v, u] += penalty
        
        # Check time occasionally (n is 100, so this loop is fast)
        if (i % 20 == 0) and (time.time() > deadline):
            return perturbed

    return perturbed
# EVOLVE-BLOCK-END
