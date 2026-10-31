import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: if we're close to the limit, fall back to fast NN
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        c1 = unvisited_nodes[0]
        c2 = unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1) if cost1 <= cost2 else int(c2)
    
    # Distances from current node to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    # Distances from destination to all unvisited
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]  # shape (n_unvisited,)
    
    # Destination bias grows as we approach the end:
    # When few nodes remain, heavily favor nodes close to destination
    # This is a structural change: use a more aggressive bias schedule
    # that kicks in earlier and is stronger
    if n_unvisited <= 5:
        dest_weight = 0.9
    elif n_unvisited <= 10:
        dest_weight = 0.6
    elif n_unvisited <= 15:
        dest_weight = 0.4
    elif n_unvisited <= 25:
        dest_weight = 0.2
    else:
        dest_weight = 0.05
    
    # Base score: linear combination of distance to current and distance to destination
    # This is the core hybrid approach
    base_score = (1.0 - dest_weight) * dist_from_current + dest_weight * dist_to_dest
    
    # Isolation penalty: for each candidate i, estimate how "isolated" the remaining
    # nodes would be if we pick i. This is the refinement machinery from inspiration 3.
    # To keep this fast, we only do a full isolation check when n_unvisited is small enough.
    if n_unvisited <= 20:
        # Compute a cheap isolation score for each candidate
        # For candidate i, the remaining set is unvisited_nodes minus i.
        # Approximate: sum over remaining nodes of (min distance to another remaining node)
        # This is O(n^2) which is fine for n <= 20.
        
        # Precompute: for each node j in unvisited, the two smallest distances to other unvisited nodes
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n, n)
        min1 = np.full(n_unvisited, np.inf)
        min2 = np.full(n_unvisited, np.inf)
        
        for j in range(n_unvisited):
            row = sub_dist[j].copy()
            row[j] = np.inf
            idx = np.argmin(row)
            min1[j] = row[idx]
            row[idx] = np.inf
            min2[j] = np.min(row)
        
        # For each candidate i, compute the isolation cost
        # The isolation cost is approximated by the max of min2[j] for j != i
        # This is a safe upper bound on the new min distance after removing i
        max_min2_excl = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            # Find the max of min2 excluding position i
            if n_unvisited > 1:
                # Create a copy of min2 with position i set to -inf
                min2_excl = min2.copy()
                min2_excl[i] = -np.inf
                max_min2_excl[i] = np.max(min2_excl)
            else:
                max_min2_excl[i] = 0.0
        
        # Normalize the isolation cost to be comparable to base_score
        # Use a scaling factor that makes the isolation penalty meaningful
        # The max possible isolation cost is roughly the diameter of the instance
        # For uniform random points in [0,1]^2, the diameter is sqrt(2)
        # So we scale by a factor of 0.1 to make it a soft penalty
        isolation_penalty = 0.1 * max_min2_excl
        
        # Combine base score with isolation penalty
        total_score = base_score + isolation_penalty
    else:
        # For larger n_unvisited, just use the base score
        total_score = base_score
    
    # Pick the candidate with the minimum total score
    best_idx = np.argmin(total_score)
    return int(unvisited_nodes[best_idx])
