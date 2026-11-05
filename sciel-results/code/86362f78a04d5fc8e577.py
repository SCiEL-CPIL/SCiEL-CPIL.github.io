"""Structural change: pre-computed Clarke-Wright tour with capacity-aware splitting.
Instead of re-scoring every candidate each step, we build a global savings tour once,
then serve customers by walking along that tour, splitting routes when capacity runs out."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

# Cache: keyed by (id of distance_matrix, tuple of demands, depot)
_tour_cache = {}

def _build_savings_tour(depot, demands, distance_matrix):
    """
    Build a Clarke-Wright savings tour.
    Returns a list of nodes (customers only, not depot) in tour order.
    """
    n = len(demands)  # includes depot at index 0
    customers = list(range(1, n))
    if not customers:
        return []

    d = distance_matrix  # shorthand

    # Compute savings for all pairs: s(i,j) = d(i,depot) + d(depot,j) - d(i,j)
    # We'll build a priority queue of savings
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = d[ci, depot] + d[depot, cj] - d[ci, cj]
            if s > 0:
                savings.append((s, ci, cj))

    # Sort by savings descending
    savings.sort(key=lambda x: -x[0])

    # Union-Find for connectivity tracking
    parent = {node: node for node in customers}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # For each customer, track which "arms" of the tour they connect to
    # Each customer can have at most 2 connections (left and right in the tour)
    left_conn = {node: None for node in customers}  # neighbor on the left
    right_conn = {node: None for node in customers}  # neighbor on the right
    degree = {node: 0 for node in customers}

    for s, ci, cj in savings:
        ri, rj = find(ci), find(cj)
        if ri == rj:
            continue  # would create a cycle
        if degree[ci] >= 2 or degree[cj] >= 2:
            continue  # can't add more connections

        # Check if adding this edge would create a sub-tour that doesn't include depot
        # In CW, we merge two routes: (depot...ci) and (cj...depot) -> (depot...ci-cj...depot)
        # The standard CW algorithm: accept if it doesn't form a cycle in the "route" sense.
        # Since we're using Union-Find on customers, and each component is a path,
        # we need to ensure we're connecting at the ends (degree < 2).
        
        # Add the connection
        if degree[ci] == 0:
            left_conn[ci] = cj
        elif degree[ci] == 1:
            # ci already has one connection; the new one goes to the other side
            if left_conn[ci] is not None:
                right_conn[ci] = cj
            else:
                left_conn[ci] = cj
        if degree[cj] == 0:
            right_conn[cj] = ci
        elif degree[cj] == 1:
            if right_conn[cj] is not None:
                left_conn[cj] = ci
            else:
                right_conn[cj] = ci

        degree[ci] += 1
        degree[cj] += 1
        union(ci, cj)

    # Now reconstruct the tour. Find the start: a node with degree 1 (or 0 if isolated).
    # Actually, in a proper CW tour, all nodes should have degree 2 except possibly
    # if some savings were <= 0 (no connection). Let's find endpoints.
    
    # Find nodes with degree < 2
    endpoints = [node for node in customers if degree[node] < 2]
    
    if not endpoints:
        # All nodes have degree 2: it's a cycle. Break it at any node.
        start = customers[0]
    else:
        # Pick an endpoint as start
        start = endpoints[0]
    
    # Walk the tour
    tour = [start]
    visited = {start}
    curr = start
    while True:
        # Determine next node
        nxt = None
        if left_conn[curr] is not None and left_conn[curr] not in visited:
            nxt = left_conn[curr]
        elif right_conn[curr] is not None and right_conn[curr] not in visited:
            nxt = right_conn[curr]
        else:
            break
        if nxt is None:
            break
        tour.append(nxt)
        visited.add(nxt)
        curr = nxt
    
    # If we didn't visit all nodes, add remaining (isolated or disconnected)
    for node in customers:
        if node not in visited:
            tour.append(node)
            visited.add(node)
    
    return tour

def _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Use the pre-computed savings tour to decide the next node.
    Strategy: Among feasible unvisited nodes, pick the one that appears earliest
    in the savings tour (cyclic order). This respects the global structure.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time check: if tight, use fast nearest neighbor fallback
    if _time_remaining() < 1.0:
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    # Get or build the savings tour
    key = (id(distance_matrix), tuple(demands), depot)
    if key not in _tour_cache:
        _tour_cache[key] = _build_savings_tour(depot, demands, distance_matrix)
    tour = _tour_cache[key]
    
    if not tour:
        # Fallback to nearest
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    # Create position map: node -> index in tour
    tour_pos = {node: idx for idx, node in enumerate(tour)}
    tour_len = len(tour)
    
    # For each feasible node, compute a "forward distance" from current_node
    # along the tour. We want to continue in the tour direction.
    # Find current_node's position (or the closest if current_node is depot)
    if current_node in tour_pos:
        curr_pos = tour_pos[current_node]
    else:
        # current_node is depot or not in tour; find the feasible node closest to depot
        # that is earliest in the tour
        curr_pos = -1  # before the start
    
    # For each feasible node, compute cyclic forward distance from curr_pos
    # We want the node that is "next" in the tour direction, considering
    # that we should continue the current route.
    
    best_node = feasible[0]
    best_score = float('inf')
    
    for n in feasible:
        pos = tour_pos.get(n, tour_len)  # if not in tour, put at end
        # Cyclic forward distance from curr_pos to pos
        if curr_pos >= 0:
            fwd = (pos - curr_pos) % tour_len
        else:
            fwd = pos  # just absolute position
        
        # Combine forward tour position with distance from current node
        # We weight the tour position heavily but add a small distance penalty
        # to break ties and handle the "just started" case
        dist_from_current = distance_matrix[current_node, n]
        
        # Normalize dist to [0,1]
        max_dist = np.max(distance_matrix[current_node, feasible]) if len(feasible) > 1 else 1.0
        if max_dist == 0:
            max_dist = 1.0
        norm_dist = dist_from_current / max_dist
        
        # Score: primarily tour position, secondarily distance
        score = fwd + 0.1 * norm_dist * tour_len
        
        if score < best_score:
            best_score = score
            best_node = n
    
    return int(best_node)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        # Fallback to fast nearest feasible
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    if len(unvisited_nodes) == 0:
        return depot

    return _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
