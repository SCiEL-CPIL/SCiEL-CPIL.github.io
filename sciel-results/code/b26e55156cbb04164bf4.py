"""Structural change: Beam search with 2-step lookahead scoring and exact DP tail."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # seconds (10s safety margin before 150s kill)


def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Finds the optimal next node for the remaining set using Held-Karp DP.
    Returns the first node in the optimal sequence.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Map unvisited nodes to local indices 0..n-1
    full_mask = (1 << n) - 1
    
    INF = float('inf')
    # dp[mask][i] = min cost to start at current_node, visit nodes in mask, end at unv[i]
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base cases: start at current_node, go to unv[i]
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    for mask in range(1 << n):
        if mask == 0:
            continue
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask, i]
            if current_cost == INF:
                continue
            
            # Try extending to j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                step_cost = distance_matrix[unv[i], unv[j]]
                new_cost = current_cost + step_cost
                
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i

    # Find the best ending node i in full_mask
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        return int(unv[0])
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    path = []
    while current_mask != 0:
        path.append(current_i)
        if current_mask & (current_mask - 1) == 0:
            first_i = current_i
            break
        
        prev_i = parent[current_mask, current_i]
        if prev_i == -1:
            break
        current_mask = current_mask ^ (1 << current_i)
        current_i = prev_i
        
    if first_i == -1:
        first_i = 0
        
    return int(unv[first_i])


def _beam_search_select(current_node, destination_node, unvisited_nodes, distance_matrix, beam_width=5, lookahead_depth=3):
    """
    Beam search over partial tours with 2-step lookahead scoring.
    Expands a set of partial tours (each ending at a different node) and
    scores extensions using a combined heuristic.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    dm = distance_matrix
    
    if n <= 2:
        # Handle small cases directly
        if n == 0:
            return int(destination_node)
        if n == 1:
            return int(unv[0])
        u1, u2 = int(unv[0]), int(unv[1])
        cost1 = dm[current_node, u1] + dm[u1, u2] + dm[u2, destination_node]
        cost2 = dm[current_node, u2] + dm[u2, u1] + dm[u1, destination_node]
        return u1 if cost1 <= cost2 else u2
    
    # Precompute pairwise distances among unvisited nodes
    sub_dm = dm[np.ix_(unv, unv)].copy()
    np.fill_diagonal(sub_dm, np.inf)
    min_onward = sub_dm.min(axis=1)  # min distance from each unvisited to any other unvisited
    
    # Score function for a partial tour state: (current_node, remaining_set)
    def _score_state(node, remaining_mask):
        """Score a state by immediate + onward + destination cost."""
        rem_nodes = [unv[i] for i in range(n) if remaining_mask & (1 << i)]
        if not rem_nodes:
            return dm[node, destination_node]
        
        # Immediate cost to each remaining node
        imm_costs = dm[node, rem_nodes]
        # Onward cost from each remaining node
        onward_costs = np.array([min_onward[i] for i in range(n) if remaining_mask & (1 << i)])
        # Destination cost
        dest_costs = dm[rem_nodes, destination_node]
        
        # Combined score: for each candidate next node, estimate total path cost
        # We pick the node that minimizes: dist(node, cand) + min_onward[cand] + dest[cand]
        # This is a 3-step lookahead: current -> cand -> best_next -> destination
        two_step = imm_costs + onward_costs + dest_costs
        return np.min(two_step)
    
    # Initialize beam: single state at current_node with all nodes unvisited
    full_mask = (1 << n) - 1
    beam = [(current_node, full_mask, 0.0)]  # (node, mask, accumulated_cost)
    
    # Expand beam for lookahead_depth steps
    for depth in range(lookahead_depth):
        new_beam = []
        for node, mask, cost in beam:
            # Get available nodes
            avail = [i for i in range(n) if mask & (1 << i)]
            if not avail:
                new_beam.append((node, mask, cost + dm[node, destination_node]))
                continue
            
            # Score each available node
            scores = []
            for i in avail:
                cand = unv[i]
                imm = dm[node, cand]
                rem_mask = mask ^ (1 << i)
                # Estimate remaining cost with simple heuristic
                if rem_mask == 0:
                    rem_cost = dm[cand, destination_node]
                else:
                    # Use min onward + dest as proxy
                    rem_nodes = [unv[j] for j in range(n) if rem_mask & (1 << j)]
                    if len(rem_nodes) == 1:
                        rem_cost = dm[cand, rem_nodes[0]] + dm[rem_nodes[0], destination_node]
                    else:
                        # 2-step lookahead from cand
                        cand_sub = dm[np.ix_(rem_nodes, rem_nodes)].copy()
                        np.fill_diagonal(cand_sub, np.inf)
                        min_next = cand_sub.min(axis=1)
                        dest_from_rem = dm[rem_nodes, destination_node]
                        # Pick best next from cand
                        two_step = dm[cand, rem_nodes] + min_next + dest_from_rem
                        rem_cost = np.min(two_step)
                total = cost + imm + rem_cost
                scores.append((total, i, cand))
            
            # Keep top beam_width candidates
            scores.sort(key=lambda x: x[0])
            for total, i, cand in scores[:beam_width]:
                new_mask = mask ^ (1 << i)
                new_beam.append((cand, new_mask, cost + dm[node, cand]))
        
        # Prune beam to keep only beam_width best states
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
    
    # Pick the best state's first move from the original current_node
    # We need to trace back: the best beam state tells us the sequence of first depth nodes
    # But we only need the FIRST node to visit.
    # Reconstruct: expand from current_node, find which first node leads to best score
    best_first = None
    best_score = float('inf')
    for i in range(n):
        cand = unv[i]
        rem_mask = full_mask ^ (1 << i)
        # Quick score: immediate + estimated remaining
        imm = dm[current_node, cand]
        if rem_mask == 0:
            rem_cost = dm[cand, destination_node]
        else:
            rem_nodes = [unv[j] for j in range(n) if rem_mask & (1 << j)]
            if len(rem_nodes) == 1:
                rem_cost = dm[cand, rem_nodes[0]] + dm[rem_nodes[0], destination_node]
            else:
                cand_sub = dm[np.ix_(rem_nodes, rem_nodes)].copy()
                np.fill_diagonal(cand_sub, np.inf)
                min_next = cand_sub.min(axis=1)
                dest_from_rem = dm[rem_nodes, destination_node]
                two_step = dm[cand, rem_nodes] + min_next + dest_from_rem
                rem_cost = np.min(two_step)
        total = imm + rem_cost
        if total < best_score:
            best_score = total
            best_first = cand
    
    return int(best_first)


def _select_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Heuristic for larger n: 2-step lookahead blended with destination awareness.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    if n == 2:
        u1 = int(unv[0])
        u2 = int(unv[1])
        cost_u1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        cost_u2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        if cost_u1 <= cost_u2:
            return u1
        else:
            return u2

    dm = distance_matrix
    d_cur = dm[current_node, unv]

    sub = dm[np.ix_(unv, unv)]
    np.fill_diagonal(sub, np.inf)
    d_onward = sub.min(axis=1)

    two_step = d_cur + d_onward
    d_dest = dm[unv, destination_node]
    
    # Normalize components to [0, 1]
    d_dest_range = d_dest.max() - d_dest.min()
    if d_dest_range > 1e-12:
        d_dest_norm = (d_dest - d_dest.min()) / d_dest_range
    else:
        d_dest_norm = np.zeros_like(d_dest)

    two_step_span = two_step.max() - two_step.min()
    if two_step_span > 1e-12:
        two_step_norm = (two_step - two_step.min()) / two_step_span
    else:
        two_step_norm = np.zeros_like(two_step)

    d_cur_span = d_cur.max() - d_cur.min()
    if d_cur_span > 1e-12:
        d_cur_norm = (d_cur - d_cur.min()) / d_cur_span
    else:
        d_cur_norm = np.zeros_like(d_cur)

    # Blend: prioritize 2-step lookahead
    score = (0.60 * two_step_norm + 0.25 * d_cur_norm + 0.15 * d_dest_norm)
    idx = int(np.argmin(score))
    return int(unv[idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        return int(unv[np.argmin(distance_matrix[current_node, unv])]) if n > 0 else int(destination_node)
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Use exact DP for small remaining sets
    if n <= 13 and elapsed < _DEADLINE - 5.0:
        return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    
    # Use beam search for medium n
    if n <= 25 and elapsed < _DEADLINE - 10.0:
        return _beam_search_select(current_node, destination_node, unv, distance_matrix, beam_width=5, lookahead_depth=3)
    
    # Fallback to heuristic
    return _select_heuristic(current_node, destination_node, unv, distance_matrix)
# EVOLVE-BLOCK-END
