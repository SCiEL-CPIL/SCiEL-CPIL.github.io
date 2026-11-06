"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: total wall-clock budget per evaluation instance
_DEADLINE_SECONDS = 110.0  # safety margin under 120s
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _reset_timer():
    global _start_time
    _start_time = time.time()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for Capacitated Vehicle Routing: pick next customer or return to depot.
    
    Strategy: Spatial clustering + regret-based selection.
    - Score each feasible candidate by a combination of:
      1. Immediate distance from current node
      2. Regret: how much worse the second-best option is (opportunity cost)
      3. Local density: how many other feasible nodes are nearby (cluster cohesion)
      4. Capacity utilization: prefer filling the vehicle before closing
    - Return to depot when capacity is low or no feasible node exists.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = _elapsed()
    if elapsed > _DEADLINE_SECONDS:
        # Safety: return depot to close route
        return 0
    
    if len(unvisited_nodes) == 0:
        return 0
    
    # Filter feasible nodes (demand fits in remaining capacity)
    feasible_mask = np.array([demands[n] <= rest_capacity for n in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # Distances from current node to all feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # If remaining capacity is very low (only 1-2 more small demands fit), prefer nearest
    # to minimize wasted capacity
    if rest_capacity <= 10:
        return int(feasible_nodes[np.argmin(dists_from_current)])
    
    # Compute regret: for each feasible node i, regret = min_dist_to_second_best - min_dist_to_i
    # i.e., how much worse we'd be if we didn't pick i
    # More precisely: regret_i = (min over j != i of dist(current, j)) - dist(current, i)
    # But a better regret: for each node i, look at what's the best alternative if we skip i
    # We approximate: sort distances, regret of the k-th closest is (min of others) - dist_k
    
    # Efficient regret computation
    # For each node, the "opportunity cost" of not visiting it is how close it is compared to others
    # Use: regret_i = min_{j != i} dist_j - dist_i  (if i is the closest, regret is high)
    
    min_dist = np.min(dists_from_current)
    second_min_dist = np.partition(dists_from_current, 1)[1]
    
    # Regret score: higher means more urgent to visit (uniquely well-positioned)
    # If node i is the unique closest, regret = second_min - dist_i (large positive)
    # If node i is far, regret is negative (less urgent)
    regret = second_min_dist - dists_from_current
    
    # Local density: for each feasible node, count how many other feasible nodes are within radius
    # Use a soft density measure: sum of exp(-dist^2 / sigma^2) to nearby feasible nodes
    # This rewards nodes in dense clusters
    
    # Compute pairwise distances among feasible nodes
    if n_feasible > 1:
        # Subsample for speed if too many
        if n_feasible > 50:
            # Use all but with a faster approximation
            idx = np.random.choice(n_feasible, size=min(n_feasible, 50), replace=False)
            sample_nodes = feasible_nodes[idx]
            sample_dists = distance_matrix[current_node, sample_nodes]
            # For density, use distance to depot as a proxy for location
            # Actually, let's just use the distance matrix directly
            pair_dists = distance_matrix[np.ix_(sample_nodes, sample_nodes)]
            sigma2 = 2.0 * np.median(np.diag(pair_dists)) ** 2 if np.median(np.diag(pair_dists)) > 0 else 1.0
            density = np.exp(-pair_dists / max(sigma2, 1.0))
            np.fill_diagonal(density, 0)
            local_density = density.sum(axis=1)
            # Map back: only sampled nodes have density, others get 0
            full_density = np.zeros(n_feasible)
            full_density[idx] = local_density
        else:
            pair_dists = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
            diag = np.diag(pair_dists).copy()
            np.fill_diagonal(pair_dists, 0)
            sigma2 = 2.0 * np.median(diag[diag > 0]) ** 2 if np.any(diag > 0) else 1.0
            density = np.exp(-pair_dists / max(sigma2, 1.0))
            np.fill_diagonal(density, 0)
            full_density = density.sum(axis=1)
    else:
        full_density = np.array([0.0])
    
    # Normalize components
    dist_norm = dists_from_current / (np.max(dists_from_current) + 1e-9)
    regret_norm = regret / (np.max(regret) - np.min(regret) + 1e-9)
    density_norm = full_density / (np.max(full_density) + 1e-9)
    
    # Combined score: minimize distance, maximize regret urgency, maximize density
    # Weight: distance 0.4, regret 0.35, density 0.25
    score = 0.4 * dist_norm - 0.35 * regret_norm - 0.25 * density_norm
    
    best_idx = np.argmin(score)
    best_node = feasible_nodes[best_idx]
    
    # Check elapsed time
    if _elapsed() > _DEADLINE_SECONDS - 5:
        return int(best_node)
    
    return int(best_node)
# EVOLVE-BLOCK-END
