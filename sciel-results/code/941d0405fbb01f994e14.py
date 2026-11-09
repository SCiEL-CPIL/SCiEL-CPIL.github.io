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
    Select the next node using a 1-step lookahead heuristic.
    
    The score for a candidate node c is:
    dist(current, c) + (1 - w) * min_dist(c, other_unvisited) + w * dist(c, destination)
    
    where w = 1 / len(unvisited_nodes).
    
    This avoids creating "spikes" (long edges) by ensuring the next edge from c is also short,
    and blends in the closing edge cost as the tour nears completion.
    """
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        if len(unvisited_nodes) == 0:
            return int(destination_node)
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    candidates = unvisited_nodes
    n = len(candidates)

    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])

    # Vectorized calculation
    # 1. Distance from current to all candidates
    d_curr = distance_matrix[current_node, candidates]
    
    # 2. Distance from all candidates to destination
    d_dest = distance_matrix[candidates, destination_node]
    
    # 3. Minimum distance from each candidate to any OTHER candidate
    #    Create NxN distance matrix among candidates
    cand_idx = candidates.astype(int)
    cand_dist = distance_matrix[np.ix_(cand_idx, cand_idx)]
    
    # Set diagonal to infinity so we don't pick self
    np.fill_diagonal(cand_dist, np.inf)
    
    # Min distance to any other unvisited node
    d_next_min = np.min(cand_dist, axis=1)
    
    # 4. Combine scores
    # w is the weight for the destination edge. 
    # As n decreases, w increases, making the closing edge more important.
    # Early on (large n), w is small, so we focus on local connectivity (d_next_min).
    w = 1.0 / n
    
    # We want to minimize the sum.
    # If n==2, w=0.5. 
    # Score = d_curr + 0.5 * d_next_min + 0.5 * d_dest
    # Note: if n==2, d_next_min is just the distance to the other candidate.
    # This is equivalent to checking the path current -> c1 -> c2 -> dest vs current -> c2 -> c1 -> dest.
    # Actually, if we pick c1, next is c2, then dest.
    # Cost if pick c1: d(curr, c1) + d(c1, c2) + d(c2, dest)
    # Cost if pick c2: d(curr, c2) + d(c2, c1) + d(c1, dest)
    # My formula:
    # Score(c1) = d(curr, c1) + 0.5 * d(c1, c2) + 0.5 * d(c1, dest)
    # Score(c2) = d(curr, c2) + 0.5 * d(c2, c1) + 0.5 * d(c2, dest)
    # This is a weighted average, not the exact future cost. 
    # However, for n=2, the exact future cost after picking c1 is d(c1, c2) + d(c2, dest).
    # My heuristic uses 0.5 * d(c1, c2) + 0.5 * d(c1, dest).
    # This might not be optimal for n=2.
    
    # Let's refine the weight or the term.
    # A better approximation for the future cost after picking c is:
    # If n > 1: min_{v != c} dist(c, v)
    # If n == 1: dist(c, dest)
    
    # Actually, if we pick c, the *next* node will be chosen by the same heuristic.
    # So the "lookahead" should ideally simulate the next choice.
    # But for n=2, the next choice is forced to be the other one.
    # So for n=2, the cost of picking c is d(curr, c) + d(c, other) + d(other, dest).
    # We can compute this exactly for small n.
    
    if n == 2:
        c0 = candidates[0]
        c1 = candidates[1]
        # Path if pick c0: curr -> c0 -> c1 -> dest
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        # Path if pick c1: curr -> c1 -> c0 -> dest
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        if cost0 < cost1:
            return int(c0)
        else:
            return int(c1)
            
    # For n > 2, use the heuristic
    # d_next_min is the min dist to other unvisited.
    # We blend d_next_min and d_dest.
    # As n gets smaller, d_dest becomes more critical.
    # Let's use a stronger weight for d_dest when n is small.
    # w = 1.0 / n is linear.
    # Maybe w = 1.0 / (n - 1) ? No, n=3 -> w=0.5.
    
    # Let's stick to the weighted sum but ensure the transition is smooth.
    # The previous logic: Score = d_curr + (1-w)*d_next_min + w*d_dest
    # Let's try to make it slightly more aggressive on the destination when n is small.
    # w = 1.0 / (n * 0.5) ? No.
    
    # Let's just use the standard lookahead. It is structurally sound.
    # To improve, we can normalize the terms? No, they are in same units.
    
    scores = d_curr + (1.0 - w) * d_next_min + w * d_dest
    
    idx = np.argmin(scores)
    return int(candidates[idx])
