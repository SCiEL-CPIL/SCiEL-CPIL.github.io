import numpy as np
import time
from itertools import permutations

_start_time = None

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def _brute_force_tail(current, dest, unvisited, dist_mat):
    """
    Find the best next node to pick such that the optimal completion of the rest is minimized.
    Total cost = dist(current, c) + optimal_path(c, rest, dest)
    """
    n = len(unvisited)
    if n == 0:
        return 0, dest
    if n == 1:
        c = unvisited[0]
        cost = dist_mat[current, c] + dist_mat[c, dest]
        return cost, c
        
    # If n is small, we can compute the optimal cost for each candidate c
    # by solving the TSP on the remaining nodes starting from c and ending at dest.
    # This is still exponential. For n=8, 7! = 5040. For 8 candidates, 8 * 5040 = 40k. OK.
    # For n=10, 9! = 362k. 10 * 362k = 3.6M. Might be slow in Python.
    # Let's limit brute force to n <= 7.
    
    best_cost = float('inf')
    best_c = unvisited[0]
    
    for i, c in enumerate(unvisited):
        cost_to_c = dist_mat[current, c]
        remaining = np.delete(unvisited, i)
        
        # Solve TSP for: start=c, visit all in remaining, end=dest
        # This is a Hamiltonian path problem.
        # If len(remaining) == 0:
        #   cost = cost_to_c + dist_mat[c, dest]
        # Else:
        #   We need min path c -> perm(remaining) -> dest
        
        if len(remaining) == 0:
            total = cost_to_c + dist_mat[c, dest]
        else:
            # Use DP or Brute Force if very small
            # For simplicity and speed, if remaining > 4, use a greedy estimate
            # If remaining <= 4, brute force
            if len(remaining) <= 4:
                min_tail = float('inf')
                for perm in permutations(remaining):
                    curr = c
                    cost = 0
                    for node in perm:
                        cost += dist_mat[curr, node]
                        curr = node
                    cost += dist_mat[curr, dest]
                    if cost < min_tail:
                        min_tail = cost
                total = cost_to_c + min_tail
            else:
                # Greedy estimate for tail
                # Nearest neighbor from c through remaining to dest
                curr = c
                rem = list(remaining)
                cost = 0
                while rem:
                    # Find nearest in rem
                    d = dist_mat[curr, rem]
                    idx = np.argmin(d)
                    cost += d[idx]
                    curr = rem[idx]
                    rem.pop(idx)
                cost += dist_mat[curr, dest]
                total = cost_to_c + cost

        if total < best_cost:
            best_cost = total
            best_c = c
            
    return best_cost, int(best_c)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
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

    # Use Brute Force / Exact for small N
    if n <= 7:
        _, best_c = _brute_force_tail(current_node, destination_node, candidates, distance_matrix)
        return best_c

    # For larger N, use 2-step Lookahead
    # Score = dist(current, c) + dist(c, nearest_neighbor(c))
    
    # Precompute submatrix for candidates
    cand_dist_sub = distance_matrix[np.ix_(candidates, candidates)]
    
    scores = np.zeros(n)
    
    for i in range(n):
        d1 = distance_matrix[current_node, candidates[i]]
        
        # Find nearest neighbor of i in candidates
        # Exclude self
        row = cand_dist_sub[i].copy()
        row[i] = np.inf
        d2 = np.min(row)
        
        # Combine
        # Weight for lookahead: 0.5 seems reasonable
        scores[i] = d1 + 0.5 * d2
        
    idx = np.argmin(scores)
    return int(candidates[idx])
