"""Global Clustering Heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a global clustering strategy.
    1. If time is running out or no nodes, return depot.
    2. Filter feasible nodes by capacity.
    3. If few feasible nodes, use a combined savings/nearest-neighbor score (fast path).
    4. Otherwise, perform a lightweight global clustering of feasible nodes.
       - Form clusters based on spatial proximity (simplified Voronoi/K-Means logic).
       - Identify the cluster that best aligns with the current route direction and capacity.
       - Select the node within that cluster closest to the current node (or best savings).
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]
    
    if len(feasible) == 0:
        return 0

    # Fast path for small sets or late game: use a refined greedy score
    # This avoids overhead of clustering when there are few choices.
    if len(feasible) < 15:
        d_cur_dep = distance_matrix[current_node, depot]
        d_dep_cand = distance_matrix[depot, feasible]
        d_cur_cand = distance_matrix[current_node, feasible]
        
        # Savings: d(cur,dep) + d(dep,c) - d(cur,c)
        savings = d_cur_dep + d_dep_cand - d_cur_cand
        
        # Nearest Neighbor: -d(cur,c)
        nn_score = -d_cur_cand
        
        # Blend: 70% savings, 30% nearest neighbor to avoid zig-zag
        # Normalize scores to comparable scales? 
        # Savings can be large, NN is distance. 
        # Let's just maximize savings, but penalize if it's very far from current?
        # Actually, for small sets, pure savings is usually fine, but let's add a slight NN bias.
        # score = savings - 0.1 * d_cur_cand
        score = savings - 0.1 * d_cur_cand
        best_idx = np.argmax(score)
        return int(feasible[best_idx])

    # Global Clustering Strategy
    # We want to group nodes that are likely to be visited in the same route segment.
    # A simple spatial clustering:
    # 1. Get coordinates? We only have distance matrix.
    # 2. We can approximate "clusters" by finding groups of nodes that are close to each other.
    
    # Let's use a "Greedy Cluster Formation" approach:
    # 1. Pick a seed node from feasible (e.g., the one with highest savings or closest to current).
    # 2. Grow the cluster by adding nearest neighbors until capacity or size limit.
    # 3. Repeat for remaining nodes.
    # 4. Choose the cluster that has the best "entry cost" from current_node.
    
    # However, doing full clustering every step is O(N^2) or worse.
    # Let's optimize:
    # We only need to decide the NEXT node.
    # Insight: The next node should be part of a "dense" group of nodes that can be served within the remaining capacity.
    
    # Strategy:
    # 1. Calculate "potential" for each feasible node: 
    #    - Proximity to current node (low is good)
    #    - Proximity to other feasible nodes (high density is good -> suggests a cluster)
    #    - Capacity fit (prefer nodes that allow filling the capacity)
    
    # Let's compute a "Cluster Attraction" score for each feasible node.
    # For each feasible node i:
    #   - d_cur_i: distance from current to i
    #   - avg_dist_to_others: average distance from i to other feasible nodes (low = dense)
    #   - savings: standard savings
    
    # This is O(K^2) where K is number of feasible nodes. K <= 50. 2500 ops is fast.
    
    K = len(feasible)
    # Get sub-matrix of distances between feasible nodes
    # distance_matrix is N x N. We need rows/cols for 'feasible'
    # dm_feas = distance_matrix[np.ix_(feasible, feasible)]
    
    # To save memory/time, we can just use the full matrix indexing.
    # d_feas_feas = distance_matrix[np.ix_(feasible, feasible)]
    
    # Let's compute average distance to other feasible nodes for each node in feasible.
    # This indicates how "clustered" the node is.
    # Low avg dist -> part of a dense cluster -> likely to be visited in a sequence.
    
    # Vectorized calculation:
    # d_cur_feas = distance_matrix[current_node, feasible] (1, K)
    # d_feas_dep = distance_matrix[depot, feasible] (1, K)
    
    d_cur_feas = distance_matrix[current_node, feasible]
    d_feas_dep = distance_matrix[depot, feasible]
    
    savings = distance_matrix[current_node, depot] + d_feas_dep - d_cur_feas
    
    # Calculate density score:
    # For each node i in feasible, calculate mean distance to all other nodes in feasible.
    # We can do this by summing distance_matrix[feasible][:, feasible] for each row, 
    # but excluding self-distance (which is 0).
    
    # dm_sub = distance_matrix[np.ix_(feasible, feasible)]
    # sum_dist = np.sum(dm_sub, axis=1)
    # mean_dist = sum_dist / (K - 1) if K > 1 else 0
    
    # To be efficient, let's just compute this if K is not too large.
    # K is at most 50. 50x50 matrix creation is trivial.
    dm_sub = distance_matrix[np.ix_(feasible, feasible)]
    if K > 1:
        sum_dist = np.sum(dm_sub, axis=1)
        mean_dist = sum_dist / (K - 1)
    else:
        mean_dist = np.array([0.0])
    
    # We want to MINIMIZE mean_dist (high density).
    # We want to MAXIMIZE savings.
    # We want to MINIMIZE d_cur_feas (close to current).
    
    # Normalize scores to [0, 1] to blend them effectively.
    def norm(x):
        min_x = np.min(x)
        max_x = np.max(x)
        if max_x == min_x:
            return np.zeros_like(x)
        return (x - min_x) / (max_x - min_x)
    
    # Higher is better for savings.
    score_savings = norm(savings)
    
    # Lower is better for mean_dist (density). So invert.
    score_density = 1.0 - norm(mean_dist)
    
    # Lower is better for d_cur_feas. So invert.
    score_proximity = 1.0 - norm(d_cur_feas)
    
    # Blend:
    # High weight on savings (route structure), medium on density (clustering), medium on proximity (local step).
    # Let's try: 0.5 * savings + 0.3 * density + 0.2 * proximity
    total_score = 0.5 * score_savings + 0.3 * score_density + 0.2 * score_proximity
    
    best_idx = np.argmax(total_score)
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
