"""Model-and-optimize TSP constructive heuristic.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, eta, delta) via a
     small grid search using a fast approximate tour builder.
  2. Use a vectorized multi-step lookahead (2-step + 3-step) for efficiency.
  3. Add a smooth connectivity/isolation penalty term.

Score per candidate c:
  score(c) = d(cur,c) + alpha*d(c,dest) + gamma*min_{u in rem\\{c}} d(c,u)
             + eta*min_{v in rem\\{c}} [d(c,v) + min_{w in rem\\{c,v}} d(v,w)]
             + delta * (d(c,dest) - min_u d(u,dest))  [isolation penalty]
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses a structural 3-step lookahead heuristic:
    1. For each candidate c, estimate the cost of the next 2 moves (c -> n -> m -> dest).
    2. This is done by pre-computing the best 2-step completion cost for every node n.
    3. This avoids parameter fitting and uses exact local minimization.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    rem_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # 1. Compute distance from current to all candidates
    d_cur = distance_matrix[current_node, rem_arr]
    
    # 2. Compute sub-matrix of distances between unvisited nodes
    # D_U[i, j] = distance between rem_arr[i] and rem_arr[j]
    D_U = distance_matrix[np.ix_(rem_arr, rem_arr)]
    
    # 3. Compute distance from each unvisited node to destination
    d_dest = distance_matrix[rem_arr, destination_node]
    
    if n_unvisited == 2:
        # If only 2 nodes left, we must visit both.
        # Order: cur -> c1 -> c2 -> dest OR cur -> c2 -> c1 -> dest
        # But we return one node. The caller will call us again for the last one.
        # We should pick the one that leads to the better total cost.
        # Cost 1: d(cur, c1) + d(c1, c2) + d(c2, dest)
        # Cost 2: d(cur, c2) + d(c2, c1) + d(c1, dest)
        # Since d(c1, c2) == d(c2, c1), we compare:
        # d(cur, c1) + d(c2, dest) vs d(cur, c2) + d(c1, dest)
        # Wait, the "next" node is just one. The cost of picking c1 is d(cur, c1).
        # The subsequent cost is d(c1, c2) + d(c2, dest).
        # Total if pick c1: d(cur, c1) + d(c1, c2) + d(c2, dest)
        # Total if pick c2: d(cur, c2) + d(c2, c1) + d(c1, dest)
        # We want to minimize the total tour length.
        # So we should pick c1 if (d(cur, c1) + d(c1, c2) + d(c2, dest)) < (d(cur, c2) + d(c2, c1) + d(c1, dest))
        # Note: d(c1, c2) is the same.
        # So compare: d(cur, c1) + d(c2, dest) vs d(cur, c2) + d(c1, dest)
        
        cost_pick_0 = d_cur[0] + D_U[0, 1] + d_dest[1]
        cost_pick_1 = d_cur[1] + D_U[1, 0] + d_dest[0]
        
        if cost_pick_0 < cost_pick_1:
            return int(rem_arr[0])
        else:
            return int(rem_arr[1])
    
    # General case: n_unvisited >= 3
    
    # Compute H(n) = min_{m in U, m != n} ( d(n, m) + d(m, dest) )
    # This is the cost of going from n to m and then closing the tour to dest.
    # M[n, m] = d(n, m) + d(m, dest)
    M = D_U + d_dest[np.newaxis, :]
    
    # Mask diagonal to ensure m != n
    np.fill_diagonal(M, np.inf)
    
    # H(n) is the min over m for each row n
    H = np.min(M, axis=1)
    
    # Now, for each candidate c, we want to pick n in U, n != c
    # that minimizes d(c, n) + H(n)
    # L[c, n] = d(c, n) + H(n)
    L = D_U + H[np.newaxis, :]
    
    # Mask diagonal to ensure n != c
    np.fill_diagonal(L, np.inf)
    
    # Score(c) = d(cur, c) + min_{n != c} L[c, n]
    min_lookahead = np.min(L, axis=1)
    
    scores = d_cur + min_lookahead
    
    best_idx = int(np.argmin(scores))
    return int(rem_arr[best_idx])

# EVOLVE-BLOCK-END
