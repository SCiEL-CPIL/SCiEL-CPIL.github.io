import numpy as np
import time
from itertools import permutations

# Global deadline management
_start_time = None
_DEADLINE_SECONDS = 115.0  # Safety margin below 120s limit

def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_exceeded():
    global _start_time
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE_SECONDS

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # If time is running out, fall back to a very fast heuristic
    if _time_exceeded():
        d_curr = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(d_curr)
        return int(unvisited_nodes[best_idx])

    U = unvisited_nodes
    n = len(U)
    
    # Precompute distances from current and to destination
    d_curr = distance_matrix[current_node, U]
    d_dest = distance_matrix[destination_node, U]
    
    # Submatrix of distances between unvisited nodes
    sub = distance_matrix[np.ix_(U, U)]
    
    # If the set is small enough, we can afford a more exhaustive search
    # n! grows fast. 6! = 720, 7! = 5040.
    # We use a hybrid approach:
    # 1. If n is small (<= 7), do exact search.
    # 2. If n is larger, use a smarter heuristic with lookahead.
    
    if n <= 7:
        best_cost = float('inf')
        best_first_node = -1
        
        # Optimization: We only care about the first node in the permutation.
        # We can iterate over the first node, and for the rest, we don't need
        # full permutations if we were doing dynamic programming, but for n=7
        # permutations are fine.
        # Actually, for n=7, 5040 permutations is trivial.
        
        for p in permutations(U):
            # Path: current -> p[0] -> ... -> p[-1] -> destination
            # We can break early if the partial sum exceeds best_cost
            cost = distance_matrix[current_node, p[0]]
            if cost >= best_cost:
                continue
            for i in range(n - 1):
                cost += distance_matrix[p[i], p[i+1]]
                if cost >= best_cost:
                    break
            else:
                cost += distance_matrix[p[-1], destination_node]
                if cost < best_cost:
                    best_cost = cost
                    best_first_node = p[0]
                    
        return int(best_first_node)

    # For larger n, use a 2-step or 3-step lookahead heuristic
    # We want to minimize: d(current, u) + d(u, v) + ... + d(last, dest)
    
    # Heuristic 1: 2-step lookahead
    # Score(u) = d(current, u) + min_{v in U\{u}} (d(u, v) + d(v, dest))
    # Note: min_{v in U\{u}} (d(u, v) + d(v, dest)) is not simply d(u, dest) or min d(u, v).
    # It's the cost of the best 2-node path from u to dest through another unvisited node.
    # However, the remaining nodes after u and v are still unvisited, so this is an approximation.
    
    # Let's refine the heuristic.
    # A common effective heuristic is:
    # Score(u) = d(current, u) + alpha * d(u, dest) + beta * min_{v != u} d(u, v)
    
    # Let's try a 2-step lookahead that considers the exit cost more carefully.
    # For each candidate u, calculate the cost of going to u, then to the best next v, then to dest?
    # No, we have n-2 nodes left after u and v.
    
    # Let's stick to a robust 2-step lookahead:
    # 1. Calculate d_curr for all U.
    # 2. For each u in U, find the best v in U\{u} to go to next.
    #    The cost of u -> v is d(u, v).
    #    The "exit cost" from v is approximated by d(v, dest).
    #    So, look_2(u) = min_{v in U\{u}} (d(u, v) + d(v, dest))
    #    Score(u) = d(current, u) + look_2(u)
    
    # Vectorized calculation of look_2
    # We need min over v != u of sub[u, v] + d_dest[v]
    
    # Create a matrix where M[u, v] = sub[u, v] + d_dest[v]
    # Then set diagonal to inf, and take min over axis=1.
    
    M = sub + d_dest[None, :]  # Broadcast d_dest to each row
    np.fill_diagonal(M, np.inf)
    
    look_2 = np.min(M, axis=1)
    
    scores = d_curr + look_2
    
    # Add a small tie-breaker or refinement?
    # Sometimes pure 2-step is not enough.
    # Let's check if we can afford a 3-step for slightly larger n?
    # 3-step: min_{v != u} (d(u, v) + min_{w != u, v} (d(v, w) + d(w, dest)))
    # This is O(n^3) which is fine for n=50 (125k ops).
    
    # Let's implement 3-step lookahead for better accuracy.
    # Cost(u) = d(current, u) + min_{v != u} [ d(u, v) + min_{w != u, v} (d(v, w) + d(w, dest)) ]
    
    # Step 1: Calculate look_2_from_v_to_dest for all v
    # look_2_from_v_to_dest[v] = min_{w != v} (d(v, w) + d(w, dest))
    # This is exactly the `look_2` we calculated above, but wait.
    # In `look_2` above, we calculated min_{v != u} (d(u, v) + d(v, dest)).
    # So `look_2[u]` is the cost of the best 2-node path from u to dest.
    
    # Now for 3-step:
    # We want min_{v != u} (d(u, v) + look_2[v])
    # Let M3[u, v] = sub[u, v] + look_2[v]
    # Set diagonal to inf.
    # look_3[u] = min_{v != u} M3[u, v]
    
    M3 = sub + look_2[None, :]
    np.fill_diagonal(M3, np.inf)
    look_3 = np.min(M3, axis=1)
    
    # Score for 3-step lookahead
    scores_3 = d_curr + look_3
    
    # Which heuristic is better?
    # 2-step is O(n^2). 3-step is O(n^2) as well (since look_2 is precomputed).
    # 3-step should be at least as good as 2-step in terms of lookahead depth.
    # Let's use 3-step.
    
    best_idx = np.argmin(scores_3)
    return int(U[best_idx])
