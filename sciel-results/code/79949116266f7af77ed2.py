"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Args:
        distance_matrix: (n, n) symmetric distance matrix
        coordinates: (n, 2) coordinates
        demands: (n,) demands (node 0 is depot with demand 0)
        capacity: vehicle capacity
        
    Returns:
        (n, n) heuristic matrix where larger = more attractive
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    d = distance_matrix.copy().astype(np.float64)
    np.fill_diagonal(d, np.inf)
    
    # Ensure finite distances
    d = np.where(np.isfinite(d), d, 1e10)
    
    # Simulate ACO dynamics with cooling schedule to create a static heuristic
    # that encodes expected pheromone reinforcement trajectory
    
    # Multiple virtual "phases" representing different stages of the search
    # Early phase: high alpha (distance-focused), late phase: low alpha (demand-aware)
    phases = [
        {'alpha': 3.0, 'demand_weight': 0.0, 'weight': 0.3},  # Early: pure distance
        {'alpha': 2.0, 'demand_weight': 0.2, 'weight': 0.4},  # Mid: balanced
        {'alpha': 1.5, 'demand_weight': 0.5, 'weight': 0.3},  # Late: demand-aware
    ]
    
    demand_ratio = demands / capacity  # (n,)
    H = np.zeros((n, n))
    
    for phase in phases:
        alpha = phase['alpha']
        dw = phase['demand_weight']
        w = phase['weight']
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # Inverse distance with phase-specific exponent
        base = 1.0 / (d ** alpha)
        
        # Demand-awareness: penalize edges to high-demand nodes
        # From depot: mild penalty (fresh capacity)
        # From customer: stronger penalty (uncertain remaining capacity)
        depot_penalty = 1.0 - 0.1 * dw * demand_ratio
        customer_penalty = 1.0 - 0.4 * dw * demand_ratio
        
        H_phase = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    H_phase[i, j] = 1e-9
                    continue
                if i == 0:
                    penalty = max(depot_penalty[j], 0.1)
                else:
                    penalty = max(customer_penalty[j], 0.1)
                H_phase[i, j] = base[i, j] * penalty
        
        H += w * H_phase
    
    # Normalize to avoid numerical issues
    H_max = np.max(H)
    if H_max > 0:
        H = H / H_max
    
    # Ensure all entries are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
