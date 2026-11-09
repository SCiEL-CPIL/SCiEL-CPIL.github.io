"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    
    We use an inverse distance power law: eta_ij = 1 / (d_ij + eps)^beta.
    Based on observations, beta=3 provided a significant improvement over beta=1.
    We test beta=4 to further sharpen the preference for short edges, which helps
    in forming compact vehicle routes in CVRP.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    # Add a small epsilon to avoid division by zero for depot-to-depot or self-loops
    # though the ant logic handles unvisited, d_ii=0 is possible.
    eps = 1e-9
    beta = 4.0
    
    # Compute the heuristic matrix
    # Using np.power for efficiency
    heuristic = 1.0 / np.power(distance_matrix + eps, beta)
    
    # Ensure no non-finite values (though with eps and positive distances, this should be safe)
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # Since distances are non-negative, 1/(d+eps)^beta is always positive.
    
    return heuristic
# EVOLVE-BLOCK-END
