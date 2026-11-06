"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline safety
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded 145s (5s buffer before 150s limit)
    if time.time() - _start_time > 145:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Identify feasible nodes (demand fits in remaining capacity)
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible node, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # --- Spatial Density Clustering Strategy ---
    # Goal: Pick the node that is in a dense cluster of remaining demand.
    # This encourages filling the truck with nearby customers, reducing
    # the total distance traveled per unit of demand.
    
    # 1. Calculate distances from current node to all feasible candidates
    # d_current = distance_matrix[current_node, feasible_nodes]
    
    # 2. Calculate distances between all feasible candidates
    # We need a distance submatrix for feasible_nodes x feasible_nodes
    # dist_sub[i, j] = distance between feasible_nodes[i] and feasible_nodes[j]
    
    # Efficiently extract submatrix
    # distance_matrix is likely a 2D array (n x n)
    # We can index it with advanced indexing
    # dist_sub = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # To avoid creating a large temporary array if feasible_nodes is large (but max 50, so 50x50=2500 is fine),
    # we proceed with the submatrix.
    
    # Optimization: Pre-compute distance from depot to feasible nodes? 
    # Not strictly needed for density, but might help for tie-breaking or closing logic.
    
    # 3. Define a radius for "locality"
    # The problem size is 50 customers. Distances are likely normalized or Euclidean.
    # We need a radius that captures a few neighbors. 
    # A dynamic radius based on the k-th nearest neighbor might be robust.
    # Let's use a radius that is the median distance from current node to feasible nodes, 
    # or a fixed fraction of the max distance.
    # Better: Use a k-nearest neighbor approach. 
    # For each candidate, look at the demand of other feasible nodes within distance R.
    
    # Let's estimate a reasonable radius R.
    # Calculate pairwise distances among feasible nodes.
    dist_sub = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each candidate i, count the demand of nodes j (j != i) within distance R.
    # We need to choose R. 
    # Let's try R = 1.0 * (median distance from current to feasible nodes)?
    # Or simply look at the top-K nearest neighbors in terms of density.
    
    # Alternative simpler metric: 
    # Score(i) = d(current, i) - lambda * (Sum of demands of feasible nodes within distance d(current, i) * factor)
    # This is getting complex. Let's stick to the "density in Voronoi" idea simplified.
    
    # Simplified Density Score:
    # For each candidate i:
    #   density_score = sum(demand[j] for j in feasible_nodes if j != i and dist(i, j) < R)
    #   total_score = d(current, i) - alpha * density_score
    
    # How to pick R? 
    # Let's pick R such that on average, each node has ~3 neighbors in the feasible set.
    # Or simply use the distance to the 2nd nearest neighbor among feasible nodes.
    
    # Let's compute distances from each feasible node to all others.
    # dist_sub is (F x F) where F is number of feasible nodes.
    # For each row i, sort the distances (excluding self).
    # Let R_i be the distance to the 2nd nearest neighbor (or 3rd if F>3).
    # This adapts R to local density.
    
    # However, using a global R is often more stable for this heuristic.
    # Let's estimate a global R based on the instance scale.
    # Max distance in matrix?
    # max_dist = np.max(distance_matrix)
    # min_dist = np.min(distance_matrix[distance_matrix > 0])
    # R = 0.2 * max_dist ?
    
    # Let's try a fixed R relative to the current-to-candidate distance.
    # Actually, let's use a "k-nearest neighbor density" where k=3.
    # For each candidate i, find the 3 closest feasible nodes (excluding i).
    # Sum their demands. This is a robust local density measure.
    
    F = len(feasible_nodes)
    if F < 2:
        # Already handled F==1, so F>=2 here
        pass
    
    # Create a mask to ignore self-loops (distance 0)
    # We want the sum of demands of the K closest nodes.
    
    # dist_sub has 0 on diagonal.
    # We can set diagonal to a large value to exclude self from "closest".
    dist_for_knn = dist_sub.copy()
    np.fill_diagonal(dist_for_knn, np.inf)
    
    # For each candidate i, we want the sum of demands of the K closest nodes.
    # Let K = min(3, F-1)
    K = min(3, F - 1)
    
    # Sort distances for each row
    # sorted_dists = np.sort(dist_for_knn, axis=1)
    # The first K columns are the distances to the K closest neighbors.
    # We need the indices to sum the demands.
    
    # Getting indices from sort is expensive if done naively for each row, but F<=50.
    # idx_sorted = np.argsort(dist_for_knn, axis=1)
    # neighbor_indices = idx_sorted[:, :K]
    
    # Then for each i, sum demands[feasible_nodes[neighbor_indices[i]]]
    
    # Let's do it vectorized-ish.
    idx_sorted = np.argsort(dist_for_knn, axis=1)
    neighbor_indices = idx_sorted[:, :K]
    
    # Get the actual node IDs of the neighbors
    neighbor_node_ids = feasible_nodes[neighbor_indices]
    
    # Sum demands of these neighbors
    # demands is an array indexed by node ID
    neighbor_demands = demands[neighbor_node_ids]
    density_scores = np.sum(neighbor_demands, axis=1)
    
    # Now, compute the cost to reach the candidate
    d_current = distance_matrix[current_node, feasible_nodes]
    
    # We want to minimize distance but maximize density.
    # Score = d_current - alpha * density_scores
    # We need to normalize density to be comparable to distance.
    # Max possible density is sum of demands of K nodes. Max demand is 9. So max ~ 27.
    # Distances are likely in a similar range (1-100?). 
    # Let's normalize density by the max density found in this step.
    max_density = np.max(density_scores)
    if max_density > 0:
        norm_density = density_scores / max_density
    else:
        norm_density = np.zeros_like(density_scores)
    
    # Tune alpha. 
    # If alpha is too high, we might pick a far node with high density.
    # If alpha is too low, it's just nearest neighbor.
    # Let's try alpha = 0.5.
    # Score = d_current - 0.5 * norm_density * (max_d_current - min_d_current)?
    # Better to scale density to the range of distances.
    d_range = np.max(d_current) - np.min(d_current)
    if d_range > 0:
        scaled_density = norm_density * d_range
    else:
        scaled_density = np.zeros_like(norm_density)
        
    alpha = 0.5
    scores = d_current - alpha * scaled_density
    
    # Pick the node with the minimum score
    idx = np.argmin(scores)
    best_node = feasible_nodes[idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
