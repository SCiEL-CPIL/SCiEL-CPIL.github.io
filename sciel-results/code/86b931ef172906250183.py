"""Savings-based heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a combined savings and nearest-neighbor heuristic.
    Returns 0 (depot) if no feasible node can be added or if time limit is reached.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Convert to numpy array for vectorized operations
    unvisited = np.array(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible = unvisited[demands[unvisited] <= rest_capacity]
    
    if len(feasible) == 0:
        return 0

    # 2. Calculate savings score
    # Savings = dist(current, depot) + dist(depot, candidate) - dist(current, candidate)
    # High savings means it's better to visit the candidate now than to return to depot and start a new route.
    
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_cand = distance_matrix[depot, feasible]
    d_cur_cand = distance_matrix[current_node, feasible]
    
    savings = d_cur_dep + d_dep_cand - d_cur_cand
    
    # 3. Blend savings with nearest-neighbor to reduce zig-zag patterns
    # Nearest neighbor favors close nodes, which helps cluster routes and avoid long detours.
    # We normalize both scores to [0, 1] range before blending.
    
    # Normalize savings to [0, 1]
    s_min = np.min(savings)
    s_max = np.max(savings)
    s_range = s_max - s_min
    if s_range > 1e-9:
        savings_norm = (savings - s_min) / s_range
    else:
        savings_norm = np.zeros_like(savings)
    
    # Normalize distance to [0, 1] (inverted so closer = higher score)
    d_min = np.min(d_cur_cand)
    d_max = np.max(d_cur_cand)
    d_range = d_max - d_min
    if d_range > 1e-9:
        dist_norm = 1.0 - (d_cur_cand - d_min) / d_range
    else:
        dist_norm = np.zeros_like(d_cur_cand)
    
    # Blend: 70% savings + 30% nearest neighbor
    # This prioritizes savings (cluster formation) while avoiding excessive detours
    alpha = 0.7
    combined_score = alpha * savings_norm + (1 - alpha) * dist_norm
    
    # 4. Capacity-aware penalty: penalize nodes that would leave very little remaining capacity
    # This helps avoid situations where we pick a large-demand node early and can't fit others later
    remaining_cap = rest_capacity - demands[feasible]
    # Penalize if remaining capacity is below 20% of original capacity
    # We don't have original capacity directly, so use a relative threshold
    cap_penalty = np.where(remaining_cap < 5, 0.1, 0.0)
    combined_score = combined_score - cap_penalty
    
    best_idx = np.argmax(combined_score)
    
    return int(feasible[best_idx])

# EVOLVE-BLOCK-END
