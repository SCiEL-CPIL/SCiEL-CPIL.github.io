"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for the entire evaluation run to enforce deadline
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    if time.time() - _start_time > 145:
        # Fallback to simple nearest feasible to avoid timeout
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # 1. Identify feasible nodes
    unvisited_array = np.array(unvisited_nodes)
    if len(unvisited_array) == 0:
        return depot
        
    feasible_mask = demands[unvisited_array] <= rest_capacity
    feasible_nodes = unvisited_array[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. Calculate scores for feasible nodes
    # Strategy: Savings Heuristic + Local Density Clustering
    
    # Distances from current node to feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Savings: S(i,j) = d(i,depot) + d(depot,j) - d(i,j)
    d_i_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    savings = d_i_depot + d_depot_j - dists
    
    # Local Density: Sum of inverse distances to other feasible nodes
    # This encourages picking nodes in dense clusters of feasible demand
    # O(N^2) calculation for N feasible nodes
    n_feasible = len(feasible_nodes)
    density_scores = np.zeros(n_feasible)
    
    # Optimization: If N is small, full loop. If large, maybe sample? N<=50, so full loop is fine.
    # We compute pairwise distances between feasible nodes
    # dist_matrix_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    # density_scores = np.sum(1.0 / (1.0 + dist_matrix_feasible), axis=1)
    # But we exclude self (diagonal)
    
    # Efficient vectorized calculation:
    # Get submatrix of distances between feasible nodes
    idx = feasible_nodes
    sub_dist = distance_matrix[np.ix_(idx, idx)]
    
    # Set diagonal to infinity to ignore self
    np.fill_diagonal(sub_dist, np.inf)
    
    # Calculate density: sum of 1/(1+d)
    density_scores = np.sum(1.0 / (1.0 + sub_dist), axis=1)
    
    # 3. Combine scores
    # We want to minimize distance, maximize savings, maximize density.
    # Score = dist - alpha * savings - beta * density
    
    # Tuning constants:
    # alpha=1.0 makes savings very important.
    # beta=0.1 adds a slight preference for dense areas.
    alpha = 1.0
    beta = 0.1
    
    scores = dists - alpha * savings - beta * density_scores
    
    # 4. Select best node
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
