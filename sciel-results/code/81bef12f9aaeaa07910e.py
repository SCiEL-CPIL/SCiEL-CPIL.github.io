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

def _greedy_complete(curr, dest, rem, dm, budget_steps=20):
    """
    Complete a partial tour from curr through all nodes in rem, ending at dest,
    using a 2-step lookahead greedy strategy. Returns the estimated total cost.
    """
    total_cost = 0.0
    rem = np.array(rem, dtype=int)
    m = len(rem)
    
    if m == 0:
        return dm[curr, dest]
    if m == 1:
        return dm[curr, rem[0]] + dm[rem[0], dest]
    if m == 2:
        n0, n1 = rem[0], rem[1]
        cost0 = dm[curr, n0] + dm[n0, n1] + dm[n1, dest]
        cost1 = dm[curr, n1] + dm[n1, n0] + dm[n0, dest]
        return min(cost0, cost1)
    
    steps = 0
    while len(rem) > 0:
        steps += 1
        m = len(rem)
        if m == 1:
            total_cost += dm[curr, rem[0]] + dm[rem[0], dest]
            break
        elif m == 2:
            n0, n1 = rem[0], rem[1]
            cost0 = dm[curr, n0] + dm[n0, n1] + dm[n1, dest]
            cost1 = dm[curr, n1] + dm[n1, n0] + dm[n0, dest]
            total_cost += min(cost0, cost1)
            break
        else:
            # 2-step lookahead: for each candidate next node, estimate cost of
            # going to it and then to its best neighbor among remaining
            d_curr = dm[curr, rem]
            # For each node in rem, find min distance to another node in rem
            rem_dist = dm[np.ix_(rem, rem)]
            np.fill_diagonal(rem_dist, np.inf)
            min_next = np.min(rem_dist, axis=1)
            scores = d_curr + min_next
            best_idx = np.argmin(scores)
            next_node = rem[best_idx]
            total_cost += d_curr[best_idx]
            curr = next_node
            rem = np.delete(rem, best_idx)
        
        if steps >= budget_steps:
            # Fallback: simple nearest neighbor for remaining
            while len(rem) > 0:
                d_curr = dm[curr, rem]
                best_idx = np.argmin(d_curr)
                next_node = rem[best_idx]
                total_cost += d_curr[best_idx]
                curr = next_node
                rem = np.delete(rem, best_idx)
            total_cost += dm[curr, dest]
            break
    
    return total_cost

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a lookahead heuristic with adaptive depth.
    
    For each candidate c, we estimate the cost of completing the tour from c
    to destination_node using a greedy strategy with 2-step lookahead on the
    remaining nodes. We select the candidate that minimizes the total estimated
    tour length.
    """
    # Time guard
    remaining = _get_time_remaining()
    if remaining < 3.0:
        # Fallback to simple nearest neighbor
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
    
    # For small n, do exact evaluation
    if n == 2:
        c0 = candidates[0]
        c1 = candidates[1]
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        return int(c0) if cost0 < cost1 else int(c1)
    
    # Adaptive lookahead depth based on time remaining and number of candidates
    # More time remaining -> deeper lookahead
    if remaining > 100.0:
        lookahead_budget = 15
    elif remaining > 50.0:
        lookahead_budget = 10
    elif remaining > 20.0:
        lookahead_budget = 5
    else:
        lookahead_budget = 2
    
    # For very large n, limit candidates to top-k by simple score to save time
    top_k = min(n, 15)
    if n > top_k:
        simple_scores = distance_matrix[current_node, candidates] + distance_matrix[candidates, destination_node]
        top_indices = np.argsort(simple_scores)[:top_k]
        candidates = candidates[top_indices]
        n = len(candidates)
    
    best_candidate = -1
    best_score = float('inf')
    
    for c in candidates:
        # Estimate cost of completing tour from c
        rem = np.array([x for x in unvisited_nodes if x != c], dtype=int)
        total_cost = distance_matrix[current_node, c] + _greedy_complete(c, destination_node, rem, distance_matrix, lookahead_budget)
        
        if total_cost < best_score:
            best_score = total_cost
            best_candidate = c
        
        # Check time
        if _get_time_remaining() < 5.0:
            break
    
    return int(best_candidate)
