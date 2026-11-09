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


def _mst_weight(dist_matrix):
    """
    Compute the weight of the Minimum Spanning Tree of a graph defined by a distance matrix.
    Uses Prim's algorithm. Assumes dist_matrix is a square symmetric numpy array.
    """
    n = dist_matrix.shape[0]
    if n <= 1:
        return 0.0
    
    # visited array
    visited = np.zeros(n, dtype=bool)
    # min_edge array: minimum edge weight connecting a visited node to an unvisited node
    min_edge = np.full(n, np.inf)
    
    # Start from node 0
    min_edge[0] = 0.0
    
    mst_weight = 0.0
    
    for _ in range(n):
        # Find the unvisited node with the smallest min_edge
        # np.where gives indices of unvisited nodes
        unvisited_indices = np.where(~visited)[0]
        if len(unvisited_indices) == 0:
            break
        
        # Find the index in unvisited_indices that has the min min_edge
        local_min_idx = np.argmin(min_edge[unvisited_indices])
        u = unvisited_indices[local_min_idx]
        
        # Add to MST
        mst_weight += min_edge[u]
        visited[u] = True
        
        # Update min_edge for neighbors of u
        # For all unvisited nodes v, if dist[u, v] < min_edge[v], update
        # We can vectorize this:
        # current min_edge for unvisited nodes
        # dist_matrix[u, unvisited_indices]
        # We need to update min_edge[v] = min(min_edge[v], dist_matrix[u, v])
        
        # Create a mask for unvisited nodes
        # Actually, we can just update all unvisited nodes
        for v in unvisited_indices:
            if v == u:
                continue
            if not visited[v]:
                if dist_matrix[u, v] < min_edge[v]:
                    min_edge[v] = dist_matrix[u, v]
                
    return mst_weight


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

    if n == 2:
        # Two nodes c, v. Path: current -> c -> v -> destination
        # Option 1: current -> c -> v -> dest
        cost1 = distance_matrix[current_node, candidates[0]] + distance_matrix[candidates[0], candidates[1]] + distance_matrix[candidates[1], destination_node]
        # Option 2: current -> v -> c -> dest
        cost2 = distance_matrix[current_node, candidates[1]] + distance_matrix[candidates[1], candidates[0]] + distance_matrix[candidates[0], destination_node]
        
        if cost1 < cost2:
            return int(candidates[0])
        else:
            return int(candidates[1])

    # Pre-compute distance submatrix between candidates
    dist_sub = distance_matrix[np.ix_(candidates, candidates)]
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    # Distance from current to each candidate
    dist_from_current = distance_matrix[current_node, candidates]

    best_score = float('inf')
    best_node = candidates[0]

    # Weights for the heuristic
    # As n decreases, the MST term becomes less dominant (fewer nodes to connect)
    # The destination term becomes more important as we approach the end
    if n <= 5:
        w_mst = 0.1
        w_dest = 0.5
    elif n <= 15:
        w_mst = 0.5
        w_dest = 0.2
    else:
        w_mst = 1.0
        w_dest = 0.1

    for i in range(n):
        # Candidate is candidates[i]
        
        # Cost to enter this candidate
        cost_in = dist_from_current[i]
        
        # Cost to connect this candidate to the destination (guide towards closure)
        cost_dest = w_dest * dist_to_dest[i]
        
        # Compute MST of the remaining unvisited nodes (excluding candidate i)
        # This represents the "scatter" cost of the remaining nodes
        if n == 3:
            # Remaining 2 nodes. MST is just the edge between them.
            # Indices in dist_sub: the two indices not equal to i
            rem_indices = [j for j in range(n) if j != i]
            mst_cost = dist_sub[rem_indices[0], rem_indices[1]]
        else:
            # Create submatrix of remaining nodes
            rem_indices = [j for j in range(n) if j != i]
            # Extract submatrix
            rem_dist = dist_sub[np.ix_(rem_indices, rem_indices)]
            mst_cost = _mst_weight(rem_dist)
        
        total_score = cost_in + w_mst * mst_cost + cost_dest
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidates[i]

    return int(best_node)
# EVOLVE-BLOCK-END
