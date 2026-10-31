"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Improved constructive heuristic for TSP.
    
    Uses a combination of nearest neighbor with a lookahead (2-opt style)
    and a penalty for nodes far from the destination to encourage closing
    the tour efficiently.
    """
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # If only two nodes left, pick the one that minimizes the total cost
    # of going to it and then to the destination
    if len(unvisited_nodes) == 2:
        n1, n2 = unvisited_nodes[0], unvisited_nodes[1]
        # Cost if we pick n1: dist(current, n1) + dist(n1, n2) + dist(n2, destination)
        # Cost if we pick n2: dist(current, n2) + dist(n2, n1) + dist(n1, destination)
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        if cost1 <= cost2:
            return int(n1)
        else:
            return int(n2)
    
    # For larger sets, use a weighted scoring:
    # score = distance from current + alpha * distance from destination
    # This balances local optimality (nearest neighbor) with global tour closure.
    
    # Adaptive alpha: as we get closer to finishing, weight destination more heavily
    # to avoid leaving a node that's far from destination until the end.
    n_total = len(distance_matrix)
    n_remaining = len(unvisited_nodes)
    
    # Start with moderate weight, increase as we finish
    # When many nodes remain, be more greedy (alpha small)
    # When few nodes remain, be more strategic (alpha large)
    alpha = 0.3 + 0.7 * (1 - n_remaining / n_total)
    
    # Compute scores for all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_destination = distance_matrix[destination_node, unvisited_nodes]
    
    # Combined score: lower is better
    scores = dist_from_current + alpha * dist_to_destination
    
    # Add a small lookahead: for each candidate, estimate the cost of the next step
    # We do this only for the top few candidates to keep it fast
    n_candidates = min(5, len(unvisited_nodes))
    top_indices = np.argsort(scores)[:n_candidates]
    top_nodes = unvisited_nodes[top_indices]
    
    # For each top candidate, compute a 2-step lookahead cost
    # cost = dist(current, cand) + min(dist(cand, other) + dist(other, dest)) for other in unvisited
    # But this is expensive, so we approximate with:
    # cost = dist(current, cand) + min over other unvisited of dist(cand, other)
    #       + alpha * dist(cand, dest)
    
    lookahead_costs = np.zeros(n_candidates)
    for i, cand in enumerate(top_nodes):
        # Distance from current to candidate
        d_current = distance_matrix[current_node, cand]
        # Distance from candidate to destination
        d_dest = distance_matrix[cand, destination_node]
        
        # Minimum distance from candidate to any other unvisited node
        # (excluding itself)
        if n_remaining > 1:
            mask = unvisited_nodes != cand
            d_next_min = np.min(distance_matrix[cand, unvisited_nodes[mask]])
        else:
            d_next_min = 0.0
        
        # Combined lookahead cost
        lookahead_costs[i] = d_current + 0.5 * d_next_min + alpha * d_dest
    
    # Pick the candidate with the lowest lookahead cost
    best_local_idx = np.argmin(lookahead_costs)
    return int(top_nodes[best_local_idx])
# EVOLVE-BLOCK-END
