# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using a regret-based heuristic.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        # Cost if we go to n1 first: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 first: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
    
    # For n_unvisited > 2, use regret-based heuristic
    # For each candidate i, compute the cost if i is NOT chosen (i.e., we must pick someone else)
    # Regret = cost of second-best option - cost of best option
    
    # 1. Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get submatrix of distances among unvisited nodes
    sub_dist = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    # 3. For each candidate i, compute the 2-step lookahead score
    # score_i = dist_from_current[i] + min_{j != i} sub_dist[i, j]
    np.fill_diagonal(sub_dist, np.inf)
    min_dist_to_next = np.min(sub_dist, axis=1)
    scores = dist_from_current + min_dist_to_next
    
    # 4. Compute regret: for each candidate i, regret_i = second_best_score - best_score
    # Sort scores to find the two smallest
    sorted_indices = np.argsort(scores)
    best_idx = sorted_indices[0]
    second_best_idx = sorted_indices[1]
    best_score = scores[best_idx]
    second_best_score = scores[second_best_idx]
    
    # Regret-based selection: prefer candidates where the regret is highest
    # (i.e., if we don't pick this candidate, the next best option is much worse)
    # This is equivalent to picking the candidate with the highest (second_best - score_i)
    # which is maximized when score_i is smallest (since second_best is fixed)
    
    # Actually, let's use a different approach: 
    # Regret for candidate i = min_{j != i} score_j - score_i
    # We want to maximize regret, which means we want to pick the candidate where 
    # the gap between the best alternative and this candidate is largest
    
    # For each candidate i, regret_i = min_{j != i} score_j - score_i
    # The best candidate to pick is the one with the highest regret
    
    # Compute regret for each candidate
    # For candidate i, the best alternative is the minimum score among all j != i
    # This is either the global best (if i != best_idx) or the second best (if i == best_idx)
    
    regret = np.zeros(n_unvisited)
    for i in range(n_unvisited):
        if i == best_idx:
            regret[i] = second_best_score - scores[i]
        else:
            regret[i] = best_score - scores[i]
    
    # Pick the candidate with the highest regret
    best_regret_idx = np.argmax(regret)
    return int(unvisited_nodes[best_regret_idx])

# EVOLVE-BLOCK-END
