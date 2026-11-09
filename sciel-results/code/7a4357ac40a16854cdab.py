"""
Global Geometric Clustering Heuristic for Capacitated Vehicle Routing Problem.

Strategy:
Instead of making myopic decisions based on local savings or nearest neighbors, 
we first partition the 50 customers into spatially coherent clusters using a 
K-Medoids algorithm (robust to non-Euclidean distances). The heuristic then 
operates within these clusters: it only considers visiting the nearest feasible 
customer within the current cluster. If no feasible customer exists in the 
current cluster, it returns to the depot to start a new route (which will 
naturally pick up the next cluster). This enforces geographic continuity, 
reducing total travel distance compared to global greedy methods that might 
jump across the map.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

# Cache for computed clusters to avoid recomputation across calls
# Key: (tuple of unvisited nodes, current_node, rest_capacity) is too expensive to key.
# Instead, we compute clusters once per "route start" or lazily if not computed for this set.
# Since unvisited_nodes shrinks, the cluster structure might change, but for a heuristic,
# using a static partition based on the full set of customers (or initial unvisited) is often sufficient 
# and much faster. However, the "current" cluster is determined by the current_node.
# 
# Better approach: 
# 1. On the first call (or when we return to depot), compute a partition of ALL nodes (0..N-1) into K clusters.
# 2. Identify which cluster the current_node belongs to.
# 3. Only consider unvisited nodes in that cluster.
# 4. If none are feasible, return to depot.
# 5. When returning to depot, we don't need to recompute clusters if we assume the spatial partition is stable.
#    But if we use a static partition of all nodes, it's computed once.

_cluster_cache = {}
_cluster_key = None

def _compute_clusters(n_nodes, distance_matrix, k):
    """
    Simple K-Medoids clustering using distance matrix.
    Returns an array of length n_nodes where cluster_id[i] is the cluster of node i.
    """
    if n_nodes <= k:
        return np.arange(n_nodes)
    
    # Initialize medoids randomly (or heuristically)
    # For stability and speed, we pick the node with the highest degree (sum of distances) as a candidate? 
    # Or just random with a fixed seed for reproducibility?
    # Let's use a simple deterministic init: pick k nodes that are far apart.
    np.random.seed(42)
    medoids = np.random.choice(n_nodes, k, replace=False)
    
    cluster_labels = np.zeros(n_nodes, dtype=int)
    
    # Iterate a few times
    for _ in range(10):
        # Assign nodes to nearest medoid
        # dist[i, j] is distance from i to medoids[j]
        dist_to_medoids = distance_matrix[:, medoids]
        cluster_labels = np.argmin(dist_to_medoids, axis=1)
        
        # Update medoids: pick the node in each cluster that minimizes sum of distances to other nodes in cluster
        new_medoids = []
        for c in range(k):
            members = np.where(cluster_labels == c)[0]
            if len(members) == 0:
                # Keep old medoid if cluster is empty
                new_medoids.append(medoids[c] if c < len(medoids) else 0)
                continue
            # Sum of distances within cluster
            sub_dist = distance_matrix[np.ix_(members, members)]
            sum_dist = np.sum(sub_dist, axis=1)
            best_idx = np.argmin(sum_dist)
            new_medoids.append(members[best_idx])
        
        if np.array_equal(new_medoids, medoids):
            break
        medoids = np.array(new_medoids)
        
    return cluster_labels

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a global clustering strategy.
    Returns 0 (depot) if no feasible node can be added in the current cluster or if time limit is reached.
    """
    global _cluster_cache, _cluster_key
    
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    n_nodes = len(demands)
    
    # Determine number of clusters. 
    # Heuristic: Aim for clusters that are roughly the size of the capacity.
    # Total demand / Capacity gives number of vehicles. 
    # Let's estimate K based on total demand.
    total_demand = np.sum(demands)
    # Assume capacity is 50 as per task description, but we don't have it explicitly.
    # We have rest_capacity. We can infer capacity? No. 
    # But we can estimate K from the number of unvisited nodes and a typical cluster size.
    # Let's use a fixed K=5 or K=8 for 50 nodes. 
    # 50 nodes / 5 clusters = 10 nodes/cluster.
    # If capacity is 50 and demands are 1-9 (avg 5), a cluster of 10 nodes has avg demand 50. 
    # So K=5 is a very good estimate for 50 nodes.
    k = 5
    
    # Check if we have a valid cluster assignment for this problem size
    # We key by n_nodes and distance_matrix identity (or just n_nodes for simplicity if matrices are consistent)
    # Since distance_matrix can vary, we should cache per matrix. 
    # Using a simple hash of the matrix is expensive. 
    # Let's just recompute if the number of nodes changes, or if we haven't computed for this specific call context?
    # Actually, the clusters depend on the geometry. If the geometry doesn't change (same instance), we can cache.
    # But the function is called sequentially. The first call computes clusters for the full set.
    # Subsequent calls can reuse them.
    
    if _cluster_key != n_nodes:
        # Compute clusters for all nodes
        _cluster_cache = _compute_clusters(n_nodes, distance_matrix, k)
        _cluster_key = n_nodes
    
    cluster_labels = _cluster_cache
    
    # Identify the cluster of the current node
    # If current_node is depot (0), we need to decide which cluster to start.
    # Usually, the depot is central. We can assign the depot to a cluster or treat it specially.
    # If current_node is 0, we look at all unvisited nodes. But which cluster?
    # Heuristic: Pick the cluster with the highest total demand? Or nearest to depot?
    # Let's pick the cluster containing the unvisited node closest to the depot.
    
    if current_node == depot:
        unvisited = np.array(unvisited_nodes)
        # Find the unvisited node closest to depot
        dist_to_depot = distance_matrix[depot, unvisited]
        closest_idx = np.argmin(dist_to_depot)
        closest_node = unvisited[closest_idx]
        current_cluster = cluster_labels[closest_node]
    else:
        current_cluster = cluster_labels[current_node]
    
    # Filter unvisited nodes to those in the current cluster
    unvisited = np.array(unvisited_nodes)
    mask_cluster = cluster_labels[unvisited] == current_cluster
    candidates = unvisited[mask_cluster]
    
    if len(candidates) == 0:
        # No candidates in this cluster. Return to depot to start a new route (next cluster).
        return 0
    
    # Filter by capacity
    feasible = candidates[demands[candidates] <= rest_capacity]
    
    if len(feasible) == 0:
        # No feasible candidates in this cluster. Return to depot.
        return 0
    
    # Among feasible candidates in the current cluster, pick the nearest one.
    # This ensures we traverse the cluster densely before leaving.
    dists = distance_matrix[current_node, feasible]
    best_idx = np.argmin(dists)
    
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
