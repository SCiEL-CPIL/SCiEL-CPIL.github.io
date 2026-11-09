"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_SECONDS = 115.0  # safety margin under 120s limit


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _check_deadline():
    return _get_elapsed() > _DEADLINE_SECONDS


def _mst_weight(nodes, distance_matrix):
    """Compute MST weight of a set of nodes using Prim's algorithm."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Map original node indices to local indices for faster lookup
    # But we can just use the submatrix directly
    submat = distance_matrix[np.ix_(nodes, nodes)]
    
    in_mst = [False] * n
    in_mst[0] = True
    min_dist = submat[0].copy()
    mst_weight = 0.0
    
    for _ in range(n - 1):
        # Find minimum distance node not in MST
        min_dist[in_mst] = np.inf
        u = np.argmin(min_dist)
        mst_weight += min_dist[u]
        in_mst[u] = True
        min_dist = np.minimum(min_dist, submat[u])
    
    return mst_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a hybrid heuristic that blends:
    1. MST-based global connectivity lower bound (structural)
    2. Greedy nearest-neighbor simulation (trajectory)
    3. Destination-awareness bias (progress)
    
    The blend weights shift towards the simulation as the number of
    unvisited nodes decreases, since the simulation becomes more accurate
    and computationally feasible.
    """
    # Deadline guard: if we're over budget, use fastest fallback
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Pre-compute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Determine blend weights based on problem size
    # For large n, MST is cheaper than full simulation
    # For small n, simulation is more accurate
    if n_unvisited <= 10:
        # Use full simulation with small destination bias
        w_sim = 1.0
        w_mst = 0.0
        bias_weight = 0.05
    elif n_unvisited <= 25:
        # Blend simulation and MST
        w_sim = 0.6
        w_mst = 0.4
        bias_weight = 0.03
    else:
        # Primarily MST-based for speed
        w_sim = 0.2
        w_mst = 0.8
        bias_weight = 0.02
    
    for i, candidate in enumerate(unvisited_nodes):
        cost_to_candidate = dists_from_current[i]
        dist_cand_to_dest = dists_to_dest[i]
        
        # Compute MST-based score
        remaining = np.delete(unvisited_nodes, i)
        if len(remaining) == 0:
            mst_score = cost_to_candidate + dist_cand_to_dest
        else:
            mst_cost = _mst_weight(remaining, distance_matrix)
            min_entry_from_cand = np.min(distance_matrix[candidate, remaining])
            min_exit_to_dest = np.min(distance_matrix[remaining, destination_node])
            # Lower bound: entry + MST + exit
            mst_score = cost_to_candidate + min_entry_from_cand + mst_cost + min_exit_to_dest
        
        # Compute simulation score if within budget
        sim_score = 0.0
        if w_sim > 0:
            rem_list = list(remaining)
            cur = candidate
            sim_cost = 0.0
            
            while rem_list:
                idx = np.argmin(distance_matrix[cur, rem_list])
                next_node = rem_list[idx]
                sim_cost += distance_matrix[cur, next_node]
                rem_list.pop(idx)
                cur = next_node
            
            sim_cost += distance_matrix[cur, destination_node]
            sim_score = cost_to_candidate + sim_cost
        
        # Blend scores
        if w_sim > 0 and w_mst > 0:
            total_cost = w_sim * sim_score + w_mst * mst_score
        elif w_sim > 0:
            total_cost = sim_score
        else:
            total_cost = mst_score
        
        # Add destination-awareness bias: prefer candidates that are
        # not too far from the destination (avoids long final jumps)
        # Normalize by average distance to destination for scale invariance
        if n_unvisited > 1:
            avg_dist_to_dest = np.mean(dists_to_dest)
            if avg_dist_to_dest > 1e-10:
                bias = bias_weight * (dist_cand_to_dest / avg_dist_to_dest)
            else:
                bias = 0.0
        else:
            bias = 0.0
        
        total_cost += bias
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = candidate
    
    return int(best_node)
# EVOLVE-BLOCK-END
