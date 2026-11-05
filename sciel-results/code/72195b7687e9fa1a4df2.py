"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from datetime import datetime, timedelta

# Global deadline guard
_EVAL_START = datetime.now()
DEADLINE = _EVAL_START + timedelta(seconds=145)  # Safety margin before 150s kill


def _time_remaining():
    return (DEADLINE - datetime.now()).total_seconds()


# Cache for distance matrix as numpy array
_DM_CACHE = {}


def _get_dm_array(distance_matrix, n):
    """Convert distance_matrix to numpy array, caching by id and shape."""
    key = id(distance_matrix)
    if key in _DM_CACHE:
        return _DM_CACHE[key]
    try:
        dm = np.asarray(distance_matrix, dtype=np.float64)
    except Exception:
        dm = np.array(distance_matrix, dtype=np.float64)
    _DM_CACHE[key] = dm
    return dm


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for selecting the next node in a capacitated vehicle routing problem.
    Uses a savings-based lookahead approach with capacity constraints.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Check if we're running out of time
    if _time_remaining() < 1.0:
        # Fallback to simple nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        dm = _get_dm_array(distance_matrix, len(demands))
        dists = dm[current_node, feasible]
        return int(feasible[np.argmin(dists)])
    
    # Find feasible nodes (can fit in remaining capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return int(depot)
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    dm = _get_dm_array(distance_matrix, len(demands))
    
    # Vectorized distances
    dist_from_current = dm[current_node, feasible]
    dist_to_depot = dm[depot, feasible]
    
    if len(feasible) <= 30:
        # For small sets, do proper 2-step lookahead with soft-min
        scores = []
        for idx, i in enumerate(feasible):
            dist_ci = dist_from_current[idx]
            # Find soft-min of distances from i to other feasible nodes
            if len(feasible) > 1:
                other_idx = [j for j in range(len(feasible)) if j != idx]
                dists_from_i = dm[i, feasible[other_idx]]
                # Soft-min with temperature
                beta = 10.0
                log_sum_exp = np.log(np.sum(np.exp(-beta * dists_from_i))) / beta
                min_next = -log_sum_exp
            else:
                min_next = 0.0
            # Combine: prefer close to current, and close to next
            scores.append(dist_ci + 0.3 * min_next)
        best_idx = np.argmin(scores)
        return int(feasible[best_idx])
    else:
        # For larger sets, use a faster approximation with depot bias
        scores = dist_from_current + 0.1 * dist_to_depot
        best_idx = np.argmin(scores)
        return int(feasible[best_idx])
# EVOLVE-BLOCK-END
