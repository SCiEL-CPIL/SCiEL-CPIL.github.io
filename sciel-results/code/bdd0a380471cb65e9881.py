"""Savings-based heuristic with multi-candidate search for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a multi-candidate savings heuristic.
    Generates several candidate nodes and evaluates them using a lookahead score.
    Returns 0 (depot) if no feasible node should be visited.
    """
    if time.time() > _DEADLINE:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)
    
    # 1. Filter by capacity
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited], dtype=bool)
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. If at depot, pick the nearest feasible node (standard)
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])

    # 3. Generate candidate scores
    # We want to balance "savings" (cluster efficiency) with "distance" (proximity)
    # and potentially "demand" (urgency).
    
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    savings = d_current_depot + d_depot_j - d_current_j
    
    # If the best saving is negative, it's cheaper to go back to depot
    if np.max(savings) < 0:
        return depot

    # Normalization for weighting
    max_saving = np.max(savings)
    min_saving = np.min(savings)
    if max_saving > min_saving:
        norm_savings = (savings - min_saving) / (max_saving - min_saving)
    else:
        norm_savings = np.ones_like(savings)

    max_dist = np.max(d_current_j)
    min_dist = np.min(d_current_j)
    if max_dist > min_dist:
        norm_dist = (d_current_j - min_dist) / (max_dist - min_dist)
    else:
        norm_dist = np.zeros_like(d_current_j)

    # Demands
    d_j = np.array([demands[i] for i in feasible_nodes], dtype=float)
    max_d = np.max(d_j)
    min_d = np.min(d_j)
    if max_d > min_d:
        norm_d = (d_j - min_d) / (max_d - min_d)
    else:
        norm_d = np.zeros_like(d_j)

    # Capacity tightness: when rest_capacity is small relative to remaining demand,
    # prefer nodes that leave more room (i.e., lower demand) to avoid dead-ends.
    total_remaining_demand = float(np.sum([demands[i] for i in unvisited_nodes]))
    cap_ratio = rest_capacity / max(total_remaining_demand, 1.0)
    # Penalty weight grows as capacity gets tight
    cap_penalty_weight = 0.3 * max(0.0, 1.0 - cap_ratio)

    # Strategy: score = norm_savings - norm_dist - cap_penalty * norm_d
    # High savings is good. High dist is bad. When capacity is tight, prefer low demand.
    
    candidates = []
    
    # Candidate 1: Max Savings
    idx1 = np.argmax(savings)
    candidates.append(idx1)
    
    # Candidate 2: Savings normalized - Distance normalized
    score2 = norm_savings - norm_dist
    idx2 = np.argmax(score2)
    candidates.append(idx2)
    
    # Candidate 3: Savings normalized - 0.5 * Distance normalized
    score3 = norm_savings - 0.5 * norm_dist
    idx3 = np.argmax(score3)
    candidates.append(idx3)
    
    # Candidate 4: Nearest Neighbor
    idx4 = np.argmin(d_current_j)
    candidates.append(idx4)
    
    # Candidate 5: Savings - Distance - capacity penalty on demand
    score5 = norm_savings - norm_dist - cap_penalty_weight * norm_d
    idx5 = np.argmax(score5)
    candidates.append(idx5)
    
    # Candidate 6: Savings - 0.5*Distance - capacity penalty
    score6 = norm_savings - 0.5 * norm_dist - cap_penalty_weight * norm_d
    idx6 = np.argmax(score6)
    candidates.append(idx6)
    
    # Remove duplicates
    unique_candidates = list(dict.fromkeys(candidates))
    
    if len(unique_candidates) == 0:
        return depot

    # Lookahead: Simulate the route for each candidate for a few steps
    # and estimate the total cost.
    
    best_score = float('inf')
    best_node = int(feasible_nodes[unique_candidates[0]])
    
    # Pre-compute demand array for speed
    all_demands = np.array(demands, dtype=float)
    dm = np.asarray(distance_matrix)
    
    # Simulate full route completion for each candidate using a fast
    # savings + nearest-neighbor hybrid, then compare predicted total length.
    for idx in unique_candidates:
        if time.time() > _DEADLINE:
            break
        
        node = int(feasible_nodes[idx])
        
        # State after picking this node
        curr = node
        cap = rest_capacity - all_demands[node]
        remaining = set(unvisited_nodes)
        remaining.discard(node)
        
        route_cost = dm[current_node, node]
        
        # Full greedy completion: at each step pick the feasible node that
        # maximizes savings (d_curr_depot + d_depot_j - d_curr_j), breaking
        # ties by nearest neighbor. This captures coherent direction and
        # depot-closure cost that per-step scoring misses.
        while remaining and cap > 0:
            # Vectorize feasibility check
            rem_arr = np.array(list(remaining), dtype=int)
            rem_demands = all_demands[rem_arr]
            feasible_mask = rem_demands <= cap
            feasible_rem = rem_arr[feasible_mask]
            
            if len(feasible_rem) == 0:
                break
            
            # Compute savings for all feasible remaining nodes
            d_curr_dep = dm[curr, depot]
            d_dep_f = dm[depot, feasible_rem]
            d_curr_f = dm[curr, feasible_rem]
            sav = d_curr_dep + d_dep_f - d_curr_f
            
            # Combined score: savings with a small distance penalty for tie-breaking
            # Prefer high savings; among similar savings, prefer closer nodes
            max_sav = np.max(sav)
            min_sav = np.min(sav)
            sav_range = max_sav - min_sav if max_sav > min_sav else 1.0
            norm_sav = (sav - min_sav) / sav_range
            
            max_d = np.max(d_curr_f)
            min_d = np.min(d_curr_f)
            d_range = max_d - min_d if max_d > min_d else 1.0
            norm_d = (d_curr_f - min_d) / d_range
            
            # Weight: primarily savings, secondary proximity
            combined = 0.8 * norm_sav - 0.2 * norm_d
            best_local = int(np.argmax(combined))
            best_next = int(feasible_rem[best_local])
            
            route_cost += dm[curr, best_next]
            curr = best_next
            cap -= all_demands[curr]
            remaining.discard(curr)
        
        # Cost to return to depot
        route_cost += dm[curr, depot]
        
        # We want to minimize this estimated cost
        if route_cost < best_score:
            best_score = route_cost
            best_node = node
    
    return best_node
# EVOLVE-BLOCK-END
