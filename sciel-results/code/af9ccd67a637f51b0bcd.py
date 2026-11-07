"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Set diagonal to infinity to avoid self-loops in heuristic calculation
    # The evaluator replaces non-positive with 1e-9, but we want 0 attractiveness for self-loops
    # Actually, the problem says "Heuristic entries that are not positive become 1e-9".
    # So if we return 0, it becomes 1e-9. If we return 1e-9, it stays 1e-9.
    # We want to minimize the chance of picking self-loops (though ants don't pick visited nodes, 
    # and node 0 is visited at start, so they won't pick 0 again immediately? 
    # Wait, "picks the next unvisited node". So self-loops are never picked.
    # However, the matrix shape must be correct.
    
    # Avoid division by zero
    d_safe = d.copy()
    d_safe[d_safe == 0] = 1e-9
    
    # Base heuristic: Inverse distance
    # Using 1/d is the standard "nearest neighbor" heuristic.
    # We will normalize each row so that the sum of heuristic values is 1.
    # This makes the heuristic a probability distribution.
    # Initial pheromone is 1. So initial selection probability is proportional to 
    # 1 * (1/d_ij) / sum(1/d_ik). This is exactly the probability of choosing 
    # the nearest neighbor in a greedy manner if pheromone is uniform.
    # As pheromone accumulates, it will bias the ants.
    
    eta = 1.0 / d_safe
    
    # Normalize rows so that each row sums to 1
    row_sums = eta.sum(axis=1, keepdims=True)
    
    # Avoid division by zero if a row sum is 0 (shouldn't happen with 1e-9)
    row_sums[row_sums == 0] = 1e-9
    
    eta = eta / row_sums
    
    # The problem states: "Heuristic entries that are not positive become 1e-9".
    # Our eta is positive. 
    # However, if d is very large, 1/d is small. 
    # If d is 0, we set it to 1e-9, so 1/d is 1e9. 
    # After normalization, the diagonal element will be huge relative to others?
    # Let's check: if d_ii = 0, eta_ii = 1e9. Other eta_ij ~ 1/1.0 = 1.
    # Sum ~ 1e9. eta_ii_norm ~ 1. eta_ij_norm ~ 1e-9.
    # This effectively makes the diagonal 1 and others 0. 
    # Since ants never pick visited nodes, and start at 0, they won't pick 0.
    # But for other nodes i, d_ii=0 means eta_ii is huge. 
    # When an ant is at node i, it cannot pick i. So this high value is ignored.
    # This is safe.
    
    # Ensure finite values
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
