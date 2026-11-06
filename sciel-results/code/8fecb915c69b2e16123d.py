import numpy as np
import time

_START = time.time()
_LIMIT = 150.0 - 10.0  # seconds of compute budget


def _greedy_complete(start, unvisited_nodes, rest_capacity, demands, dm):
    """
    Estimate the cost to complete the route starting from 'start',
    visiting all feasible nodes in 'unvisited_nodes' using Nearest Neighbor,
    and returning to the depot (0).
    """
    remaining = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not remaining:
        return dm[start, 0]
    
    cur = start
    cap = rest_capacity
    cost = 0.0
    # To speed up, we can just iterate. N is small (<=50).
    # We need to pick the nearest feasible node repeatedly.
    # A simple O(N^2) loop is fine.
    visited = set()
    while True:
        best_node = -1
        best_dist = float('inf')
        for n in remaining:
            if n in visited:
                continue
            if demands[n] > cap:
                continue
            d = dm[cur, n]
            if d < best_dist:
                best_dist = d
                best_node = n
        if best_node == -1:
            break
        cost += best_dist
        cur = best_node
        cap -= demands[best_node]
        visited.add(best_node)
    
    cost += dm[cur, 0]
    return cost


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    t0 = time.time()
    deadline = _START + _LIMIT

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = list(unvisited_nodes)
    dm = np.asarray(distance_matrix)

    # Feasible first nodes: demand <= rest_capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return feasible[0]

    best_score = np.inf
    best_node = feasible[0]

    for cand in feasible:
        if time.time() > deadline:
            break
        
        # Cost to reach candidate
        cost_to_cand = dm[current_node, cand]
        
        # Remaining capacity after visiting candidate
        rem_cap = rest_capacity - demands[cand]
        
        # Remaining nodes (excluding candidate)
        rem_nodes = [n for n in unvisited if n != cand]
        
        # Estimate cost to finish the route from candidate
        # We pass the current node as 'start' for the completion, but we are at 'cand' now.
        # The function _greedy_complete starts from 'start' and visits 'unvisited_nodes'.
        # So we start from 'cand' and visit 'rem_nodes'.
        est_finish_cost = _greedy_complete(cand, rem_nodes, rem_cap, demands, dm)
        
        total_cost = cost_to_cand + est_finish_cost
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = cand

    return int(best_node)
