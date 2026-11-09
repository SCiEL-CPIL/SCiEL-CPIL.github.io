"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    Select the next node using adaptive heuristics:
    - Early phase: 2-step lookahead (current -> candidate -> best next from candidate)
    - Late phase: destination-aware to minimize closing edge
    - Always: prefer candidates that keep options open (not too isolated)
    """
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])
    if n == 2:
        # Pick the one that gives shorter total: current->c + c->dest
        c0, c1 = candidates[0], candidates[1]
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, destination_node]
        return int(c0 if cost0 <= cost1 else c1)

    # Get distances from current node to all candidates
    d_current = distance_matrix[current_node, candidates]
    
    # Get distances from candidates to destination node
    d_dest = distance_matrix[candidates, destination_node]
    
    # 2-step lookahead: for each candidate, estimate cost of current->candidate->best_next
    # This is O(n^2) but n <= 48 here, so it's fine
    sub_matrix = distance_matrix[np.ix_(candidates, candidates)]
    np.fill_diagonal(sub_matrix, np.inf)
    # For each candidate, the best next step is the min distance to any other unvisited
    best_next_cost = np.min(sub_matrix, axis=1)
    
    # 2-step cost: d(current, c) + d(c, best_next_from_c)
    two_step_cost = d_current + best_next_cost
    
    # Adaptive weights based on remaining nodes
    # When n is large (early), emphasize 2-step lookahead
    # When n is small (late), emphasize destination distance
    if n > 20:
        # Early phase: mostly 2-step lookahead
        w_twostep = 0.7
        w_dest = 0.3
    elif n > 10:
        # Mid phase: balanced
        w_twostep = 0.5
        w_dest = 0.5
    else:
        # Late phase: heavily weight destination
        w_twostep = 0.3
        w_dest = 0.7
    
    # Normalize for fair combination
    twostep_max = np.max(two_step_cost) + 1e-10
    dest_max = np.max(d_dest) + 1e-10
    
    combined_score = w_twostep * (two_step_cost / twostep_max) + w_dest * (d_dest / dest_max)
    
    best_idx = np.argmin(combined_score)
    return int(candidates[best_idx])
# EVOLVE-BLOCK-END
