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
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Vectorized 2-Step Lookahead Heuristic
    # Score(c) = d(current, c) + min_{v != c} [ d(c, v) + min_{u != c, v} d(v, u) ]
    # Special handling for small n to account for the return to destination.
    
    # Pre-compute distances from current to all candidates
    dist_from_current = distance_matrix[current_node, candidates]
    
    # Extract submatrix of distances between candidates
    # dist_sub[i, j] = distance_matrix[candidates[i], candidates[j]]
    dist_sub = distance_matrix[np.ix_(candidates, candidates)]
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    if n == 2:
        # Path: current -> c -> v -> destination
        # Score(0) = d(cur, 0) + d(0, 1) + d(1, dest)
        # Score(1) = d(cur, 1) + d(1, 0) + d(0, dest)
        score0 = dist_from_current[0] + dist_sub[0, 1] + dist_to_dest[1]
        score1 = dist_from_current[1] + dist_sub[1, 0] + dist_to_dest[0]
        return int(candidates[0] if score0 < score1 else candidates[1])
        
    elif n == 3:
        # For each candidate c, the remaining 2 nodes must be visited in some order.
        # Since it's a triangle, the optimal path from c through the other two to dest is:
        # min( d(c, v) + d(v, u) + d(u, dest), d(c, u) + d(u, v) + d(v, dest) )
        # Note: d(v,u) == d(u,v).
        
        # Calculate scores for all 3 candidates vectorized
        # i=0: rem = {1, 2}
        # i=1: rem = {0, 2}
        # i=2: rem = {0, 1}
        
        scores = np.zeros(3)
        
        # i=0
        path1 = dist_sub[0, 1] + dist_sub[1, 2] + dist_to_dest[2]
        path2 = dist_sub[0, 2] + dist_sub[2, 1] + dist_to_dest[1]
        scores[0] = dist_from_current[0] + min(path1, path2)
        
        # i=1
        path1 = dist_sub[1, 0] + dist_sub[0, 2] + dist_to_dest[2]
        path2 = dist_sub[1, 2] + dist_sub[2, 0] + dist_to_dest[0]
        scores[1] = dist_from_current[1] + min(path1, path2)
        
        # i=2
        path1 = dist_sub[2, 0] + dist_sub[0, 1] + dist_to_dest[1]
        path2 = dist_sub[2, 1] + dist_sub[1, 0] + dist_to_dest[0]
        scores[2] = dist_from_current[2] + min(path1, path2)
        
        best_idx = np.argmin(scores)
        return int(candidates[best_idx])
        
    else:
        # n > 3
        # Vectorized 2-step lookahead
        # For each i (candidate c):
        #   cost_in = dist_from_current[i]
        #   cost_out = min_{j != i} ( dist_sub[i, j] + min_{k != i, j} dist_sub[j, k] )
        
        # Step 1: For each j, compute min_{k != j} dist_sub[j, k]
        # This is the nearest neighbor distance for each node in the submatrix,
        # but we need to exclude the specific 'i' when calculating cost_out for 'i'.
        # However, the term is min_{u != c, v} d(v, u).
        # Let's rewrite the loop logic for vectorization.
        
        # Let M[i, j] = dist_sub[i, j].
        # We want S[i] = min_{j != i} ( M[i, j] + min_{k != i, j} M[j, k] )
        
        # Precompute for each j, the two smallest distances to other nodes in the submatrix.
        # Let min1[j] be the smallest distance from j to any k != j.
        # Let min2[j] be the second smallest distance from j to any k != j.
        # Let argmin1[j] be the index k that gives min1[j].
        
        # Then min_{k != i, j} M[j, k] is:
        # - min2[j] if argmin1[j] == i
        # - min1[j] otherwise
        
        # We can compute min1 and min2 for all j at once.
        
        # To find min1 and min2 for each row j:
        # We can use np.partition or simply sort.
        # Since n is up to 50, sorting is fast.
        # But we can also just use argpartition.
        
        # Actually, simpler:
        # For each j, we want the min of M[j, :] excluding M[j, j] (which is 0) and potentially M[j, i].
        
        # Let's create a matrix where M[j, j] = inf to ignore self-loops.
        M = dist_sub.copy()
        np.fill_diagonal(M, np.inf)
        
        # Now for each j, find the smallest and second smallest values in M[j, :].
        # We can use np.argpartition.
        # indices = np.argpartition(M, 2, axis=1)
        # min1_idx = indices[:, 1] # index of smallest (0 is self, now inf)
        # Wait, argpartition with kth=1 gives the element that would be in position 1 if sorted.
        # So M[j, indices[j, 0]] <= M[j, indices[j, 1]] <= ...
        
        indices = np.argpartition(M, 1, axis=1)
        min1_idx = indices[:, 0]
        min1_val = M[np.arange(n), min1_idx]
        
        min2_idx = indices[:, 1]
        min2_val = M[np.arange(n), min2_idx]
        
        # Now compute S[i] for all i.
        # S[i] = min_{j != i} ( M[i, j] + (min2_val[j] if min1_idx[j] == i else min1_val[j]) )
        
        # Let's construct a matrix Cost[i, j] = M[i, j] + (min2_val[j] if min1_idx[j] == i else min1_val[j])
        # Then S[i] = min_{j != i} Cost[i, j]
        
        # Create a matrix AddCost where AddCost[i, j] is the second term.
        # AddCost[i, j] = min2_val[j] if min1_idx[j] == i else min1_val[j]
        
        # We can do this by creating a matrix of min1_val for all j,
        # then setting AddCost[min1_idx[j], j] = min2_val[j].
        
        AddCost = np.tile(min1_val, (n, 1))
        # For each j, set AddCost[min1_idx[j], j] to min2_val[j]
        AddCost[np.arange(n), min1_idx] = min2_val
        
        TotalCost = M + AddCost
        
        # We need min over j != i.
        # Set diagonal to inf.
        np.fill_diagonal(TotalCost, np.inf)
        
        S = np.min(TotalCost, axis=1)
        
        TotalScore = dist_from_current + S
        
        best_idx = np.argmin(TotalScore)
        return int(candidates[best_idx])

# EVOLVE-BLOCK-END
