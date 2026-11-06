import numpy as np
import itertools
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node 

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Time guard: track elapsed time
    start_time = time.time()
    deadline = start_time + 2.0  # 2 second budget per call

    # If the number of remaining nodes is small, solve the sub-problem exactly
    if n_unvisited <= 6:
        best_cost = np.inf
        best_next = int(unvisited_nodes[0])
        
        for perm in itertools.permutations(unvisited_nodes):
            cost = distance_matrix[current_node, perm[0]]
            for i in range(n_unvisited - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_next = int(perm[0])
                
        return best_next

    # For larger sets, use 3-step lookahead heuristic
    # Score(v) = d(current, v) + min_{w in U\{v}} [d(v, w) + min_{x in U\{v,w}} d(w, x)]
    # This is a 3-step lookahead that approximates the optimal continuation cost
    
    # Get sub-matrix of distances between unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity so we don't pick the node itself
    sub_dist_no_diag = sub_dist.copy()
    np.fill_diagonal(sub_dist_no_diag, np.inf)
    
    # For each node, find its 2 nearest neighbors (excluding itself)
    # nn1_idx[i]: index of nearest neighbor
    # nn1_dist[i]: distance to nearest neighbor
    # nn2_idx[i]: index of 2nd nearest neighbor
    # nn2_dist[i]: distance to 2nd nearest neighbor
    
    # Sort distances for each row to find top-2
    sorted_dists = np.sort(sub_dist_no_diag, axis=1)
    nn1_dist = sorted_dists[:, 0]
    nn2_dist = sorted_dists[:, 1]
    
    # For 3-step lookahead: for candidate v, the best 2-step continuation is:
    # min over w != v of [d(v,w) + min over x != v,w of d(w,x)]
    # We approximate this by: min over w != v of [d(v,w) + d(w, nn(w) if nn(w) != v else nn2(w))]
    
    # For each node w, precompute: 
    #   best_cont[w] = min over x != w of d(w, x)  [this is nn1_dist[w]]
    #   second_best_cont[w] = min over x != w, x != nn1(w) of d(w, x) [this is nn2_dist[w]]
    
    # But we need: for candidate v, for each w != v:
    #   d(v, w) + [d(w, nn1(w)) if nn1(w) != v else d(w, nn2(w))]
    
    # Get indices of nearest neighbors
    nn1_idx = np.argmin(sub_dist_no_diag, axis=1)
    
    # For each w, the best continuation cost when v is excluded:
    # if nn1_idx[w] != v_local_idx, then best is nn1_dist[w]
    # else best is nn2_dist[w]
    
    # Distance from current_node to each unvisited node
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate v (local index i), compute score:
    # score[i] = dist_from_current[i] + min over j != i of [sub_dist[i,j] + cont_cost[j, excluding i]]
    
    best_scores = np.full(n_unvisited, np.inf)
    
    for i in range(n_unvisited):
        # Check time
        if time.time() > deadline:
            break
        
        # For candidate v = unvisited_nodes[i]
        # For each w = unvisited_nodes[j] where j != i:
        #   cost = d(v, w) + [nn1_dist[j] if nn1_idx[j] != i else nn2_dist[j]]
        
        # Vectorized: for all j != i
        j_range = np.arange(n_unvisited)
        mask = j_range != i
        
        d_v_w = sub_dist[i, mask]  # distances from v to all other unvisited nodes
        
        # For each j in mask, get the continuation cost
        # cont_cost[j] = nn1_dist[j] if nn1_idx[j] != i else nn2_dist[j]
        cont_costs = np.where(nn1_idx[mask] != i, nn1_dist[mask], nn2_dist[mask])
        
        # Total cost for each w
        total_costs = d_v_w + cont_costs
        
        # Best 2-step continuation from v
        best_2step = np.min(total_costs)
        
        best_scores[i] = dist_from_current[i] + best_2step
    
    # If time ran out, fall back to 2-step lookahead
    if time.time() > deadline:
        # Fallback: 2-step lookahead
        nn_dist = nn1_dist
        scores = dist_from_current + nn_dist
        best_local_idx = np.argmin(scores)
        return int(unvisited_nodes[best_local_idx])
    
    best_local_idx = np.argmin(best_scores)
    return int(unvisited_nodes[best_local_idx])
