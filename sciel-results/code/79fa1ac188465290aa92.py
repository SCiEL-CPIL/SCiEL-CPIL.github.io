import numpy as np
import time

_start_time = None

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard
    if _get_time_remaining() < 1.0:
        # Fallback to simple NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])

    best_node = -1
    best_score = float('inf')

    # If few nodes left, compute exact cost of completing the tour
    if n <= 5:
        # We can even do a small search or just exact calculation for the remaining
        # For n=2: pick v such that d(c,v)+d(v,other)+d(other,dest) is min
        # For n=3: pick v such that d(c,v) + min(perm of others) ...
        # Let's just do a simple greedy with full lookahead for the remaining nodes
        # Since n is small, we can even do a brute force for the remaining if n is very small,
        # but n=5 is 120 perms, which is fine. But we are in a loop, so we just pick the best next step.
        # Actually, for n<=5, we can just evaluate the cost of picking each v and then
        # greedily (or optimally for small n) completing the rest.
        
        for i in range(n):
            v = unvisited_nodes[i]
            # Cost to go to v
            cost = distance_matrix[current_node, v]
            
            # Simulate the rest of the tour from v
            # We have n-1 nodes left.
            # Let's just use a simple nearest neighbor for the rest for speed, 
            # or better, since n is small, let's just add the min distance to destination 
            # if it's the last one, or min to next if not.
            # Actually, for n<=5, we can just calculate the exact optimal completion?
            # No, that's TSP again. But n is small.
            # Let's stick to the heuristic: 
            # If n-1 == 0: cost += d(v, dest)
            # If n-1 > 0: cost += min_{w in U\{v}} d(v, w) + ...
            
            # Let's refine:
            remaining = np.delete(unvisited_nodes, i)
            if len(remaining) == 0:
                total_score = cost + distance_matrix[v, destination_node]
            else:
                # Greedily pick the next best from remaining?
                # Or just use the 2-step lookahead logic on the remaining set?
                # Let's just use the same logic recursively or simply:
                # cost += min_{w in remaining} d(v, w)
                # This is the 2-step lookahead.
                dists_v = distance_matrix[v, remaining]
                cost += np.min(dists_v)
                # If there are more nodes, this underestimates.
                # But for n<=5, the difference is small.
                # To be safer, if n==2, we should add d(v, other) + d(other, dest).
                if n == 2:
                    other = remaining[0]
                    total_score = distance_matrix[current_node, v] + distance_matrix[v, other] + distance_matrix[other, destination_node]
                elif n == 3:
                    # d(c, v) + min( d(v, w1) + d(w1, w2) + d(w2, dest), d(v, w2) + d(w2, w1) + d(w1, dest) )
                    w1, w2 = remaining
                    path1 = distance_matrix[v, w1] + distance_matrix[w1, w2] + distance_matrix[w2, destination_node]
                    path2 = distance_matrix[v, w2] + distance_matrix[w2, w1] + distance_matrix[w1, destination_node]
                    total_score = distance_matrix[current_node, v] + min(path1, path2)
                else:
                    # For n > 3, stick to 2-step lookahead approximation
                    total_score = cost
                    
            if total_score < best_score:
                best_score = total_score
                best_node = v
                
        return int(best_node)

    # For n > 5, use 2-step lookahead
    for i in range(n):
        v = unvisited_nodes[i]
        cost = distance_matrix[current_node, v]
        
        # Find min distance from v to any other unvisited node
        dists_v = distance_matrix[v, unvisited_nodes]
        dists_v[i] = float('inf')
        min_next_dist = np.min(dists_v)
        
        total_score = cost + min_next_dist
        
        if total_score < best_score:
            best_score = total_score
            best_node = v
            
    return int(best_node)
