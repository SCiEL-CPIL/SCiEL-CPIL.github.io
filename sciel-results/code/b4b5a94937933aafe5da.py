import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Lookahead heuristic.
    
    For each candidate c, we simulate the completion of the tour from c to 
    destination_node using a 2-step greedy strategy on the remaining nodes.
    We select the candidate that results in the shortest total path length.
    """
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 2.0:
        if len(unvisited_nodes) == 0:
            return int(destination_node)
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        cands = unvisited_nodes
        scores = distance_matrix[current_node, cands] + distance_matrix[cands, destination_node]
        return int(cands[np.argmin(scores)])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])

    # For n == 2, exact evaluation
    if n == 2:
        c0 = candidates[0]
        c1 = candidates[1]
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        return int(c0) if cost0 < cost1 else int(c1)

    best_candidate = -1
    best_score = float('inf')

    for c in candidates:
        # Simulate completion starting from c
        curr = c
        # Remaining nodes are all unvisited except c
        rem = np.array([x for x in unvisited_nodes if x != c])
        
        # Cost to reach c from current
        total_cost = distance_matrix[current_node, curr]
        
        while len(rem) > 0:
            m = len(rem)
            if m == 1:
                next_node = rem[0]
                total_cost += distance_matrix[curr, next_node]
                total_cost += distance_matrix[next_node, destination_node]
                break
            elif m == 2:
                n0 = rem[0]
                n1 = rem[1]
                cost0 = distance_matrix[curr, n0] + distance_matrix[n0, n1] + distance_matrix[n1, destination_node]
                cost1 = distance_matrix[curr, n1] + distance_matrix[n1, n0] + distance_matrix[n0, destination_node]
                if cost0 < cost1:
                    total_cost += cost0
                else:
                    total_cost += cost1
                break
            else:
                # 2-step lookahead greedy
                # 1. Distance from curr to each node in rem
                d_curr = distance_matrix[curr, rem]
                
                # 2. For each node in rem, find min distance to any other node in rem
                #    This approximates the cost of the next edge after visiting this node
                rem_dist = distance_matrix[np.ix_(rem, rem)]
                np.fill_diagonal(rem_dist, np.inf)
                min_next = np.min(rem_dist, axis=1)
                
                # Score: cost to come here + cost to go to next best neighbor
                scores = d_curr + min_next
                
                best_idx = np.argmin(scores)
                next_node = rem[best_idx]
                
                # Add cost to move to next_node
                total_cost += d_curr[best_idx]
                
                # Update state
                curr = next_node
                rem = np.delete(rem, best_idx)
                
        if total_cost < best_score:
            best_score = total_cost
            best_candidate = c

    return int(best_candidate)
