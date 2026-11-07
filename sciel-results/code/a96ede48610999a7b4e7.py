"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of CVRP.
    
    Uses a multi-factor heuristic combining:
    1. Inverse distance (closer nodes more attractive)
    2. Savings-based component (edges that save distance vs returning to depot)
    3. Demand-aware component (prefer edges where remaining capacity allows more)
    4. Geometric alignment (prefer edges that continue in similar direction)
    """
    n = distance_matrix.shape[0]
    
    # 1. Inverse distance component
    # Use a small epsilon to avoid division by zero
    eps = 1e-9
    inv_dist = 1.0 / (distance_matrix + eps)
    
    # 2. Savings component
    # Savings(i, j) = dist(i, 0) + dist(0, j) - dist(i, j)
    d_i_0 = distance_matrix[:, 0]
    d_0_j = distance_matrix[0, :]
    savings = d_i_0[:, np.newaxis] + d_0_j[np.newaxis, :] - distance_matrix
    savings = np.maximum(savings, 0)
    
    # Normalize savings
    max_savings = np.max(savings)
    if max_savings > 0:
        norm_savings = savings / max_savings
    else:
        norm_savings = np.zeros_like(savings)
    
    # 3. Demand-aware component
    # For each edge (i, j), consider how well the demand of j fits
    # We want to prefer edges where the destination has lower demand
    # (so we can fit more customers on the route)
    # But also consider the current node's demand context
    
    # Create a demand-based attractiveness
    # Lower demand at destination is better for packing
    demand = demands.copy()
    demand[0] = 0  # Depot has no demand
    
    # For each potential move i -> j, the attractiveness should consider
    # how much capacity remains after adding j
    # This is complex to compute exactly, so use a proxy:
    # Prefer moving to nodes with lower demand
    demand_factor = 1.0 / (demand[np.newaxis, :] + 1)
    
    # 4. Geometric alignment component
    # Compute direction vectors and prefer edges that continue in similar direction
    # For each pair (i, j), compute the angle between vector (depot->i) and (i->j)
    # Smaller angle = better alignment
    
    # Get coordinates
    coords = coordinates
    depot_coords = coords[0]
    
    # Vector from depot to each node
    depot_to_node = coords - depot_coords[np.newaxis, :]
    
    # For edge i->j, vector is coords[j] - coords[i]
    # We want to measure how well this continues from depot->i direction
    
    # Compute alignment: dot product of normalized (depot->i) and (i->j)
    # Only meaningful for non-depot nodes
    
    # For each i, j pair:
    # dir_i = depot_to_node[i] / ||depot_to_node[i]||
    # dir_ij = (coords[j] - coords[i]) / ||coords[j] - coords[i]||
    # alignment = dot(dir_i, dir_ij)
    
    # This is expensive to compute for all pairs, so use a simpler approach:
    # Use the angle at node i between vectors (depot->i) and (j->i)
    # Actually, let's use a simpler geometric heuristic:
    # Prefer edges that don't create sharp turns
    
    # Compute the "turn penalty" for each edge
    # For node i (not depot), the incoming direction is from depot (or previous)
    # Since we don't know previous, use depot->i as reference
    
    # Simplified: use the component of (j-i) in the direction of (i-depot)
    # This rewards going "forward" away from depot
    
    # Compute for all i, j
    # dir_from_depot[i] = (coords[i] - coords[0]) / ||coords[i] - coords[0]||
    # forward_component[i,j] = dot(coords[j] - coords[i], dir_from_depot[i])
    
    # Only valid for i != 0
    forward_comp = np.zeros((n, n))
    for i in range(1, n):
        dir_i = depot_to_node[i]
        norm_i = np.linalg.norm(dir_i)
        if norm_i > eps:
            dir_i = dir_i / norm_i
            # For all j, compute dot((coords[j] - coords[i]), dir_i)
            to_j = coords - coords[i][np.newaxis, :]
            forward_comp[i, :] = np.sum(to_j * dir_i[np.newaxis, :], axis=1)
    
    # Normalize forward_comp to [0, 1]
    # The range depends on the geometry, so normalize
    max_fwd = np.max(forward_comp)
    min_fwd = np.min(forward_comp)
    if max_fwd > min_fwd:
        norm_fwd = (forward_comp - min_fwd) / (max_fwd - min_fwd)
    else:
        norm_fwd = np.ones_like(forward_comp) * 0.5
    
    # Combine all components
    # Weights: inv_dist (0.3), savings (0.3), demand (0.2), forward (0.2)
    w_inv = 0.3
    w_sav = 0.3
    w_dem = 0.2
    w_fwd = 0.2
    
    heuristic = (w_inv * inv_dist + 
                 w_sav * norm_savings + 
                 w_dem * demand_factor + 
                 w_fwd * norm_fwd)
    
    # Scale up to make values more distinct
    heuristic = heuristic * 10
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
