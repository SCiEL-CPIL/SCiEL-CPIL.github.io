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
    
    # Ensure finite
    d = np.where(np.isfinite(d), d, 1e10)
    
    # Compute base inverse distance
    # Use a moderate exponent to balance exploration vs exploitation
    alpha = 1.5
    base = 1.0 / (d ** alpha)
    
    # Demand-aware adjustment
    # The key insight: when an ant is at node i with some remaining capacity,
    # it should prefer nodes j where demand[j] is small relative to remaining capacity.
    # Since we don't know remaining capacity in a static heuristic, we use a proxy:
    # Penalize edges to high-demand nodes more when coming from customer nodes
    # (where capacity is already partially used) vs depot (fresh capacity).
    
    demand_ratio = demands / capacity  # (n,)
    
    # Build heuristic with demand-awareness
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 1e-9
                continue
            if not np.isfinite(d[i, j]) or d[i, j] < 1e-10:
                H[i, j] = 1e-9
                continue
            
            # Base attractiveness from distance
            val = base[i, j]
            
            # Demand penalty: higher demand at destination j is less attractive
            # Penalty is stronger when coming from a customer (less remaining capacity)
            if i == 0:
                # From depot: fresh capacity, mild penalty
                penalty = 1.0 / (1.0 + 0.3 * demand_ratio[j])
            else:
                # From customer: remaining capacity uncertain, stronger penalty
                # Scale penalty by how "full" the route might be
                penalty = 1.0 / (1.0 + 0.6 * demand_ratio[j])
            
            val *= penalty
            
            # Additional boost for short edges to low-demand nodes
            # This encourages building routes with many small stops
            if demand_ratio[j] < 0.2:  # Low demand node
                val *= 1.15
            
            H[i, j] = val
    
    # Normalize
    H_max = np.max(H)
    if H_max > 0:
        H = H / H_max
    
    # Ensure all entries are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return H
    
    # Try to refine with a simple simulated annealing-inspired adjustment
    # This adjusts the heuristic based on local neighborhood structure
    # to better anticipate the colony's pheromone trajectory
    
    # Compute a "potential" for each node based on its connectivity
    # Nodes that are well-connected (many short edges) get boosted
    row_sums = np.sum(base, axis=1)
    node_potential = 1.0 / (row_sums + 1e-10)
    
    # Blend in node potential as a secondary signal
    # This helps ants prefer moving through well-connected regions
    blend = 0.2
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i, j] *= (1.0 - blend + blend * node_potential[j])
    
    # Re-normalize
    H_max = np.max(H)
    if H_max > 0:
        H = H / H_max
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
