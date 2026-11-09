"""Direct heuristic for TSP next-node selection with adaptive multi-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Strategy:
    - For n_unvisited <= 2, use exact optimal completion.
    - For n_unvisited <= 10, use 2-step lookahead: minimize d(current, u) + min(d(u, v) + d(v, dest) for v != u).
      This is O(n^2) and gives a strong local view of the remaining path.
    - For n_unvisited > 10, use an adaptive blend:
        score(u) = d(current, u) + alpha * d(u, destination) + beta * min_dist_to_other_unvisited(u)
      where alpha increases as we get closer to completion (to avoid a long closing edge),
      and beta penalizes nodes that are isolated from the remaining set.
      
    The key structural change from the parent: instead of a hard phase switch at n=20,
    we use a smooth, continuous adaptation of the destination bias and connectivity penalty
    across the entire tour construction. Early on, we favor nearest-neighbor-like behavior
    with a mild connectivity check. Later, we strongly bias toward nodes that are close
    to both the current node and the destination, ensuring a short closing edge.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For two nodes left, pick the one that minimizes total edge cost to close the tour
        u, v = unvisited_nodes
        # Option 1: go to u then v: cost = d(current,u) + d(u,v) + d(v,dest)
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        # Option 2: go to v then u: cost = d(current,v) + d(v,u) + d(u,dest)
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    if n_unvisited <= 10:
        # 2-step lookahead: for each candidate u, compute the best completion cost
        # which is d(current, u) + min over v != u of (d(u, v) + d(v, destination))
        d_curr = distance_matrix[current_node, unvisited_nodes]
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_matrix, np.inf)
        
        # d_dest is distance from each unvisited node to destination
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # For each u, the best next step v minimizes d(u, v) + d(v, dest)
        # sub_matrix[i, j] is d(unvisited[i], unvisited[j])
        # We want for each i: min_j (sub_matrix[i, j] + d_dest[j])
        # This is a row-wise minimum of (sub_matrix + d_dest[None, :])
        completion_costs = sub_matrix + d_dest[None, :]
        min_completion = np.min(completion_costs, axis=1)
        
        # Total score for each candidate u
        scores = d_curr + min_completion
        
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
    else:
        # Adaptive blend for larger sets
        d_curr = distance_matrix[current_node, unvisited_nodes]
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # Compute min distance from each unvisited node to any other unvisited node
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_matrix, np.inf)
        min_next_dist = np.min(sub_matrix, axis=1)
        
        # Adaptive alpha: starts small, increases as we approach completion
        # When n_unvisited is large (e.g., 50), alpha is small.
        # When n_unvisited is small (e.g., 11), alpha is larger.
        # Use a smooth function: alpha = 1.0 / (1.0 + 0.1 * (n_unvisited - 10))
        # At n=10: alpha = 1.0
        # At n=50: alpha = 1.0 / (1.0 + 4.0) = 0.2
        alpha = 1.0 / (1.0 + 0.1 * (n_unvisited - 10))
        
        # Beta: connectivity penalty, also adaptive
        # We want to penalize isolated nodes more when there are many unvisited nodes
        beta = 0.3 / (1.0 + 0.05 * (n_unvisited - 10))
        
        # Normalize distances to make the blend scale-invariant
        # Use the mean distance as a scale factor
        mean_dist = np.mean(distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)])
        
        scores = d_curr + alpha * d_dest + beta * min_next_dist
        
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
