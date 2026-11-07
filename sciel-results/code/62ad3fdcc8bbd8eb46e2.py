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
    
    # Precompute pairwise demand compatibility: how well demands[i] + demands[j] fit in capacity
    # For edge i->j, the "remaining load" after visiting j depends on the route, but we can
    # approximate by penalizing pairs where combined demand is close to capacity
    demand_pair_sum = demands[:, None] + demands[None, :]  # (n, n)
    # Compatibility: 1 when pair sum is well below capacity, drops as it approaches capacity
    # Use a smooth sigmoid-like penalty
    pair_margin = capacity - demand_pair_sum  # positive if fits, negative if not
    pair_compat = 1.0 / (1.0 + np.exp(-0.5 * (pair_margin / capacity)))  # smooth step
    
    # Depot proximity bias: prefer nodes closer to depot for initial assignments
    depot_coords = coordinates[0]
    dist_to_depot = np.sqrt(np.sum((coordinates - depot_coords) ** 2, axis=1))  # (n,)
    dist_to_depot = np.where(dist_to_depot > 0, dist_to_depot, 1.0)
    depot_proximity = 1.0 / dist_to_depot  # higher for nodes closer to depot
    depot_proximity = depot_proximity / np.max(depot_proximity)
    
    # Spatial coherence: for edges between customers, prefer those that maintain
    # a coherent direction (low angle change)
    # Compute bearing from each node to every other
    dx = coordinates[None, :, 0] - coordinates[:, None, 0]  # (n, n)
    dy = coordinates[None, :, 1] - coordinates[:, None, 1]  # (n, n)
    angles = np.arctan2(dy, dx)  # (n, n)
    
    # For spatial coherence, we want edges that don't cause sharp turns
    # Approximate by measuring how aligned consecutive edges are
    # Simpler: penalize edges that go "against the flow" relative to depot
    # Use a smooth surrogate: cosine of angle between edge direction and depot-to-node direction
    # For each pair (i, j), compute angle between vector (i->j) and (depot->j)
    # This encourages routes that approach nodes from the depot side (outward spiral)
    depot_to_j = coordinates - depot_coords  # (n, 2)
    i_to_j = coordinates[None, :, :] - coordinates[:, None, :]  # (n, n, 2)
    
    # Compute cosine of angle between i_to_j and depot_to_j for each (i, j)
    dot_prod = np.sum(i_to_j * depot_to_j[None, :, :], axis=2)  # (n, n)
    mag_ij = np.sqrt(np.sum(i_to_j ** 2, axis=2))  # (n, n)
    mag_depot_j = np.sqrt(np.sum(depot_to_j ** 2, axis=1))  # (n,)
    mag_depot_j = mag_depot_j[None, :]  # (1, n)
    
    cos_angle = dot_prod / (mag_ij * mag_depot_j + 1e-10)  # (n, n)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # Spatial coherence score: higher when angle is small (aligned)
    spatial_coherence = 0.5 * (1 + cos_angle)  # range [0, 1]
    
    # Simulate ACO dynamics with cooling schedule to create a static heuristic
    # that encodes expected pheromone reinforcement trajectory
    
    # Multiple virtual "phases" representing different stages of the search
    # Early phase: high alpha (distance-focused), late phase: low alpha (demand-aware)
    phases = [
        {'alpha': 3.0, 'demand_weight': 0.0, 'spatial_weight': 0.1, 'weight': 0.3},  # Early: pure distance
        {'alpha': 2.0, 'demand_weight': 0.2, 'spatial_weight': 0.3, 'weight': 0.4},  # Mid: balanced
        {'alpha': 1.5, 'demand_weight': 0.5, 'spatial_weight': 0.5, 'weight': 0.3},  # Late: demand+spatial
    ]
    
    demand_ratio = demands / capacity  # (n,)
    H = np.zeros((n, n))
    
    for phase in phases:
        alpha = phase['alpha']
        dw = phase['demand_weight']
        sw = phase['spatial_weight']
        w = phase['weight']
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # Inverse distance with phase-specific exponent
        base = 1.0 / (d ** alpha)
        
        # Pairwise demand compatibility: penalize edges where combined demand is high
        # This is a structural change from per-node to per-edge demand awareness
        pair_penalty = pair_compat ** dw
        
        # Depot proximity bias: boost edges to nodes that are closer to depot
        # This helps with initial route construction
        depot_bias = 1.0 + 0.2 * dw * depot_proximity[None, :]
        
        # Spatial coherence: boost edges that maintain directional consistency
        spatial_bias = 1.0 + sw * spatial_coherence
        
        # Combine all factors
        H_phase = base * pair_penalty * depot_bias * spatial_bias
        
        # Set diagonal to small value
        np.fill_diagonal(H_phase, 1e-9)
        
        H += w * H_phase
    
    # Normalize to avoid numerical issues
    H_max = np.max(H)
    if H_max > 0:
        H = H / H_max
    
    # Ensure all entries are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
