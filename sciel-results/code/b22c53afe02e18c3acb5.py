"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _check_deadline(safety_margin=5.0):
    """Return True if we should stop and return depot."""
    elapsed = time.time() - _get_start_time()
    return elapsed > (120.0 - safety_margin)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return 0

    # Deadline guard
    if _check_deadline():
        return 0

    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands_arr = np.asarray(demands, dtype=np.float64)

    # 1. Filter feasible nodes: demand[node] <= rest_capacity
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands_arr[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if len(feasible_nodes) == 0:
        return 0

    feasible_np = np.array(feasible_nodes)
    n_feasible = len(feasible_np)

    # Structural change: Density-weighted centroid clustering approach.
    # Instead of pairwise savings or step-by-step lookahead, we model the routing
    # decision as maintaining high local spatial density. We compute a weighted
    # centroid of all feasible unvisited nodes (weighted by inverse demand, so
    # smaller demands pull the centroid more) and score each candidate by how
    # well it maintains cluster coherence:
    #   score = dist(current, candidate) + lambda * dist(candidate, centroid)
    # This prioritizes nodes that are close to the current position AND close
    # to the center of mass of the remaining feasible cluster, reducing intra-
    # cluster travel variance and avoiding "outlier" picks that fragment the
    # remaining work into multiple sub-clusters.

    # Compute density-weighted centroid of feasible nodes.
    # Weights: inverse demand (smaller demand = higher weight, more "spread" influence)
    # But to avoid division by zero, use 1/(demand + 0.5)
    weights = 1.0 / (demands_arr[feasible_np] + 0.5)
    weights /= weights.sum()
    
    # Use distance matrix to approximate positions. Since we don't have explicit
    # coordinates, we use the distance matrix itself. The centroid in a metric
    # space can be approximated by finding the point that minimizes weighted
    # distance to all feasible nodes. For efficiency, we use the feasible node
    # with minimum weighted distance as a proxy centroid, or compute a weighted
    # average of distances.
    
    # Alternative: Use the depot-relative distances to define a "position" proxy.
    # dist_to_depot[i] gives a 1D projection. We can use this to find a central
    # feasible node.
    dist_to_depot = dm[feasible_np, depot]
    
    # Weighted median-like approach: find the feasible node closest to the
    # weighted "center" in terms of distance-to-depot.
    weighted_center_depot = np.sum(weights * dist_to_depot)
    centroid_proxy_idx = np.argmin(np.abs(dist_to_depot - weighted_center_depot))
    centroid_node = feasible_np[centroid_proxy_idx]
    
    # Score each candidate:
    #   base_cost = dist(current, candidate)
    #   cluster_cost = dist(candidate, centroid_node) * lambda
    #   capacity_penalty = demand(candidate) / rest_capacity * mu
    #   total_score = base_cost + cluster_cost + capacity_penalty
    
    dist_current = dm[current_node, feasible_np]
    dist_to_centroid = dm[feasible_np, centroid_node]
    
    # Lambda: weight for cluster coherence. Higher = more emphasis on staying
    # near the cluster center.
    # Mu: weight for capacity preservation. Higher = prefer smaller demands
    #     to keep more capacity for future nodes.
    lambda_cluster = 0.3
    mu_capacity = 0.1
    
    demand_frac = demands_arr[feasible_np] / max(rest_capacity, 1e-9)
    
    scores = dist_current + lambda_cluster * dist_to_centroid + mu_capacity * demand_frac
    
    # Also consider the cost of returning to depot now
    return_cost = dm[current_node, depot]
    
    min_score = np.min(scores)
    min_idx = np.argmin(scores)
    
    # Decide: return to depot or pick the best candidate?
    # Return to depot if:
    # 1. No feasible nodes (already handled)
    # 2. Capacity is very tight and the best candidate's cost is significantly
    #    higher than just returning (risk of getting stuck with no feasible
    #    continuation)
    if rest_capacity <= 3 and min_score > return_cost * 1.5:
        return 0
    
    # Also return if the best candidate's demand would leave very little capacity
    # AND the remaining unvisited nodes are far (indicating we'd need multiple
    # routes anyway, so closing this one is better)
    best_demand = demands_arr[feasible_np[min_idx]]
    remaining_cap_after = rest_capacity - best_demand
    if remaining_cap_after <= 2 and n_feasible > 5:
        # Check if there are other feasible nodes that could fit in the
        # remaining capacity. If not, closing now is better.
        can_fit_more = any(demands_arr[f] <= remaining_cap_after for f in feasible_nodes if f != feasible_np[min_idx])
        if not can_fit_more:
            return 0
    
    return int(feasible_np[min_idx])
# EVOLVE-BLOCK-END
