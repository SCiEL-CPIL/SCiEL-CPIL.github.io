"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin from 150s

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _START_TIME, _TIME_LIMIT
    
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Convert to numpy array for fast operations
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    
    # 1. Check capacity constraints
    feasible_mask = np.array([demands[j] <= rest_capacity for j in unvisited])
    
    if not np.any(feasible_mask):
        return depot

    # Get feasible nodes
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 1:
        return int(feasible_nodes[0])
    
    # Compute distances
    dist_from_current = distance_matrix[current_node, feasible_nodes].astype(np.float64)
    dist_from_depot = distance_matrix[depot, feasible_nodes].astype(np.float64)
    
    # --- Component 1: Savings ---
    # Savings = dist(depot, next) - dist(current, next)
    # Positive savings means visiting next saves distance vs going back to depot
    savings_score = dist_from_depot - dist_from_current
    
    # --- Component 2: 2-Regret (gap to second-best) ---
    # For each node i, compute the gap between its distance and the second-closest
    # alternative. Higher gap = more regret if we skip this node.
    # Use sorted distances for efficiency
    sorted_dists = np.sort(dist_from_current)
    min_dist = sorted_dists[0]
    second_min_dist = sorted_dists[1] if n_feasible >= 2 else min_dist
    
    # Regret for node i: how much worse the best alternative is compared to node i
    # If node i is the closest, regret = second_min - min_dist (high regret to skip)
    # If node i is far, regret is low
    # General formula: regret_i = min_{j != i} dist[j] - dist[i]
    # For the closest node: regret = second_min - min
    # For others: regret = min - dist[i] (negative if farther than min)
    
    # Efficient computation: for each i, the best alternative is:
    # - if i is the argmin, best_alt = second_min_dist
    # - else, best_alt = min_dist
    argmin_idx = np.argmin(dist_from_current)
    best_alt = np.full(n_feasible, min_dist)
    best_alt[argmin_idx] = second_min_dist
    regret_raw = best_alt - dist_from_current
    
    # Normalize regret
    max_regret = np.max(regret_raw)
    min_regret = np.min(regret_raw)
    if max_regret > min_regret:
        regret_normalized = (regret_raw - min_regret) / (max_regret - min_regret)
    else:
        regret_normalized = np.zeros(n_feasible)
    
    # --- Component 3: Cluster density (proximity to other feasible nodes) ---
    # Lower average distance to other feasible nodes = better cluster fit
    # This encourages picking nodes in dense clusters, reducing future detours
    if n_feasible <= 80:
        # Compute pairwise distances among feasible nodes
        dist_among = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)].astype(np.float64)
        # Set diagonal to 0 (self-distance)
        np.fill_diagonal(dist_among, 0.0)
        # Average distance to other feasible nodes
        avg_dist_to_others = np.sum(dist_among, axis=1) / (n_feasible - 1)
    else:
        # Fallback: use distance to current as proxy
        avg_dist_to_others = dist_from_current.copy()
    
    # Normalize: lower avg_dist = better, so we invert
    max_cluster = np.max(avg_dist_to_others)
    min_cluster = np.min(avg_dist_to_others)
    if max_cluster > min_cluster:
        cluster_score = 1.0 - (avg_dist_to_others - min_cluster) / (max_cluster - min_cluster)
    else:
        cluster_score = np.ones(n_feasible)
    
    # --- Normalize savings ---
    max_savings = np.max(savings_score)
    min_savings = np.min(savings_score)
    if max_savings > min_savings:
        savings_normalized = (savings_score - min_savings) / (max_savings - min_savings)
    else:
        savings_normalized = np.zeros(n_feasible)
    
    # --- Dynamic weighting based on capacity ---
    total_capacity = 50
    capacity_ratio = rest_capacity / total_capacity
    
    # Smooth transition: as capacity decreases, shift weight toward regret and cluster
    # High capacity: emphasize savings (global optimization)
    # Low capacity: emphasize regret + cluster (local completion)
    
    # Use a smooth sigmoid-like transition
    # At capacity_ratio = 1.0: savings_weight ~ 0.60, regret ~ 0.25, cluster ~ 0.15
    # At capacity_ratio = 0.0: savings_weight ~ 0.30, regret ~ 0.45, cluster ~ 0.25
    
    t = capacity_ratio  # 0 to 1
    # Smooth interpolation
    savings_weight = 0.30 + 0.30 * t       # 0.30 to 0.60
    regret_weight = 0.45 - 0.20 * t        # 0.45 to 0.25
    cluster_weight = 0.25 - 0.10 * t       # 0.25 to 0.15
    
    # Normalize weights to sum to 1
    weight_sum = savings_weight + regret_weight + cluster_weight
    savings_weight /= weight_sum
    regret_weight /= weight_sum
    cluster_weight /= weight_sum
    
    # --- Combined score ---
    combined_score = (savings_weight * savings_normalized + 
                      regret_weight * regret_normalized + 
                      cluster_weight * cluster_score)
    
    # Small penalty for very long distances to avoid extreme detours
    max_dist = np.max(dist_from_current) if len(dist_from_current) > 0 else 1.0
    if max_dist > 0:
        dist_penalty = 0.02 * (dist_from_current / max_dist)
        combined_score = combined_score - dist_penalty
    
    # We want to maximize the combined score
    best_idx = np.argmax(combined_score)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)

# EVOLVE-BLOCK-END
