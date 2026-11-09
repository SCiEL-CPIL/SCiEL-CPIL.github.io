# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a global connectivity-aware heuristic.
    
    For each candidate next node, we estimate the cost of completing the tour by:
    1. The immediate edge from current_node to the candidate.
    2. The cost to connect the remaining unvisited nodes (after removing the candidate)
       into a path that starts from the candidate and ends at destination_node.
       We approximate this by the cost of a Minimum Spanning Tree (MST) on the remaining
       nodes plus the two "end" edges (from candidate to its best neighbor in the MST-like
       structure, and from destination_node to its best neighbor in the remaining set).
       A simpler and faster approximation: use the MST weight of the remaining unvisited
       nodes as a proxy for the path completion cost, and add the edge from current_node
       to candidate and from candidate to destination_node (or from the last unvisited to
       destination_node).
    
    A more practical and fast heuristic:
    For each candidate c in unvisited_nodes:
        - cost_direct = d(current_node, c)
        - After choosing c, the remaining nodes are unvisited_nodes without c.
        - The path must go from c through all remaining nodes to destination_node.
        - Approximate the cost of that sub-path by:
            d(c, best_neighbor_in_remaining) + MST_weight(remaining) + d(destination_node, best_neighbor_in_remaining)
          But this is expensive to compute for each candidate.
    
    A faster approximation that captures connectivity:
    - For each candidate c, compute:
        immediate_cost = d(current_node, c)
        # Estimate the "penalty" for isolating c from the rest:
        # The remaining nodes must form a path from c to destination_node.
        # A lower bound on the cost to connect the remaining nodes is the MST of the
        # remaining nodes. But we also need to connect c and destination_node to this
        # structure.
        #
        # Simpler: use the sum of the two smallest edges from c to the remaining nodes
        # and from destination_node to the remaining nodes, plus the MST of the remaining
        # nodes. But MST is O(n^2) per candidate, which is too slow.
    
    Practical fast heuristic:
    - For each candidate c:
        score = d(current_node, c) 
              + min(d(c, u) for u in unvisited_nodes if u != c)   # cost to leave c
              + min(d(destination_node, u) for u in unvisited_nodes if u != c)  # cost to enter destination
              + 0.5 * sum of the two smallest distances from c to any unvisited node (to estimate local connectivity)
    
    Actually, a well-known improvement over nearest neighbor is to use a "lookahead" 
    that considers the cost of the edge from the candidate to the best subsequent node,
    and the cost from the destination to the best remaining node.
    
    Let's use this scoring for each candidate c:
        score(c) = d(current_node, c) 
                 + min_{u in unvisited \ {c}} d(c, u) 
                 + min_{u in unvisited \ {c}} d(destination_node, u)
    
    This encourages choosing c that is close to current_node, has a close neighbor in the
    remaining set (so we don't get stuck), and leaves the destination with a close 
    remaining node to connect to.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current_node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, compute the min distance from that candidate to any other unvisited node
    # and the min distance from destination_node to any other unvisited node (excluding the candidate)
    
    # Precompute distances from destination_node to all unvisited nodes
    dist_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # For each candidate i, we need:
    #   min_j d(unvisited[i], unvisited[j]) for j != i
    #   min_j d(destination_node, unvisited[j]) for j != i
    
    # Compute the submatrix of distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row, find the min value (this is the min distance from that node to any other unvisited node)
    # But we need to exclude the node itself (which is 0)
    # Set diagonal to inf before taking min
    submatrix_copy = submatrix.copy()
    np.fill_diagonal(submatrix_copy, np.inf)
    min_dist_from_candidate = np.min(submatrix_copy, axis=1)
    
    # For the destination part: for each candidate i, we want min_j d(dest, unvisited[j]) for j != i
    # This is min(dist_from_dest) excluding dist_from_dest[i]
    # We can compute this by finding the two smallest values in dist_from_dest
    # If i is the index of the smallest, use the second smallest; otherwise use the smallest
    
    # Find the two smallest values and their indices in dist_from_dest
    if n_unvisited >= 2:
        sorted_indices = np.argsort(dist_from_dest)
        min1_idx = sorted_indices[0]
        min1_val = dist_from_dest[min1_idx]
        min2_val = dist_from_dest[sorted_indices[1]]
        
        # For each candidate i, the min distance from dest to any other unvisited node
        min_dist_from_dest_excl = np.where(
            np.arange(n_unvisited) == min1_idx, 
            min2_val, 
            min1_val
        )
    else:
        min_dist_from_dest_excl = np.zeros(n_unvisited)
    
    # Compute the total score for each candidate
    scores = dist_from_current + min_dist_from_candidate + min_dist_from_dest_excl
    
    # Choose the candidate with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
