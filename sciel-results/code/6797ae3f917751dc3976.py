import numpy as np
import time

_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    if not hasattr(_get_dm, 'cache'):
        _get_dm.cache = {}
    key = id(distance_matrix)
    if key not in _get_dm.cache:
        _get_dm.cache[key] = np.asarray(distance_matrix)
    return _get_dm.cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 5 seconds
    return (time.time() - _start_time) > 145

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    Uses a density-weighted centroid heuristic combined with Clarke-Wright savings
    to prioritize nodes that maintain high local spatial density.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    # Convert inputs to numpy arrays for fast indexing
    if not isinstance(unvisited_nodes, np.ndarray):
        unvisited_nodes = np.asarray(unvisited_nodes)
    if not isinstance(demands, np.ndarray):
        demands = np.asarray(demands)
        
    # If no unvisited nodes, return to depot
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    # Identify feasible nodes based on remaining capacity
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # If no feasible nodes, return to depot
    if len(feasible_nodes) == 0:
        return int(depot)
        
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    # Deadline guard: if we're running out of time, fall back to simple heuristic
    if _check_deadline():
        # Simple fallback: Clarke-Wright savings
        d_depot_current = dm[depot, current_node]
        d_depot_j = dm[depot, feasible_nodes]
        d_current_j = dm[current_node, feasible_nodes]
        savings = d_depot_current + d_depot_j - d_current_j
        best_idx = np.argmax(savings)
        return int(feasible_nodes[best_idx])
        
    # --- New Heuristic: Density-Weighted Centroid + Savings ---
    
    # 1. Compute Clarke-Wright Savings
    d_depot_current = dm[depot, current_node]
    d_depot_j = dm[depot, feasible_nodes]
    d_current_j = dm[current_node, feasible_nodes]
    savings = d_depot_current + d_depot_j - d_current_j
    
    # 2. Compute Density-Weighted Centroid
    # We use the inverse distance from the current node as a weight to define a "local cluster"
    # Nodes closer to current_node have higher weight in the centroid calculation.
    # This helps identify the "core" of the remaining feasible cluster.
    
    # Avoid division by zero if current_node is in feasible_nodes (it shouldn't be, but safe)
    # d_current_j is distance from current to each feasible node.
    # We use 1/d as weight. If d is 0, weight is inf. 
    # Since feasible_nodes are unvisited, d > 0 usually.
    
    # Normalize distances to avoid large/small numbers issues
    min_d = np.min(d_current_j)
    if min_d == 0:
        # If any node is at current position (shouldn't happen for unvisited), handle gracefully
        weights = np.ones(len(feasible_nodes))
    else:
        # Inverse distance weighting
        # Add a small epsilon to avoid division by zero if min_d is 0
        weights = 1.0 / (d_current_j + 1e-9)
        
    # Normalize weights to sum to 1 for stable centroid calculation
    sum_weights = np.sum(weights)
    if sum_weights > 0:
        weights_norm = weights / sum_weights
    else:
        weights_norm = np.ones(len(feasible_nodes)) / len(feasible_nodes)
    
    # We need coordinates to compute a geometric centroid. 
    # However, we only have the distance matrix. 
    # We can approximate the "direction" or use a projection.
    # Alternatively, we can define the centroid in the space of the distance matrix itself?
    # No, a true spatial centroid requires coordinates.
    
    # Since we don't have coordinates, we must approximate "density" differently.
    # Idea: A node is "dense" if it is close to many other feasible nodes.
    # Let's compute the average distance from each feasible node to all other feasible nodes.
    # Lower average distance = higher density.
    
    # Compute pairwise distances among feasible nodes
    # feasible_nodes is an array of indices
    # dm[feasible_nodes[:, None], feasible_nodes[None, :]] gives the submatrix
    
    # For efficiency, only do this if the number of feasible nodes is not too large
    # With 50 customers, this is small.
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible > 1:
        # Get submatrix of distances between feasible nodes
        # Note: dm is symmetric usually
        feasible_dm = dm[np.ix_(feasible_nodes, feasible_nodes)]
        
        # Compute mean distance from each node to all others (excluding self)
        # Set diagonal to 0 so it doesn't affect mean
        np.fill_diagonal(feasible_dm, 0.0)
        
        # Mean distance to other feasible nodes
        mean_dist_to_others = np.mean(feasible_dm, axis=1)
        
        # Lower mean_dist_to_others means higher density
        # We want to prefer nodes with lower mean_dist_to_others
        # So, density_score = -mean_dist_to_others
        
        density_score = -mean_dist_to_others
        
        # Normalize density_score to be comparable to savings
        # Savings can be positive or negative. Density score is negative.
        # Let's scale them.
        
        # Normalize savings to [0, 1]
        min_savings = np.min(savings)
        max_savings = np.max(savings)
        if max_savings > min_savings:
            savings_norm = (savings - min_savings) / (max_savings - min_savings)
        else:
            savings_norm = np.zeros_like(savings)
            
        # Normalize density_score to [0, 1]
        min_density = np.min(density_score)
        max_density = np.max(density_score)
        if max_density > min_density:
            density_norm = (density_score - min_density) / (max_density - min_density)
        else:
            density_norm = np.zeros_like(density_score)
            
        # Combine scores
        # We want to maximize both savings and density (lower mean dist)
        # Weighting: Savings is a proven strong heuristic. Density is a refinement.
        # Let's try 0.6 * savings + 0.4 * density
        
        combined_score = 0.6 * savings_norm + 0.4 * density_norm
        
        # Add a small bias towards higher demand to break ties and prefer filling capacity
        demand_bias = 0.01 * demands[feasible_nodes]
        
        final_scores = combined_score + demand_bias
        
        best_idx = np.argmax(final_scores)
        return int(feasible_nodes[best_idx])
    else:
        # Fallback if only one node (handled above) or something went wrong
        best_idx = np.argmax(savings)
        return int(feasible_nodes[best_idx])
