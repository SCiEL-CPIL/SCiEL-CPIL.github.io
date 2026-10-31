"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit

def _check_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a minimax-inspired heuristic combined with nearest neighbor.
    
    The strategy aims to minimize the maximum edge length in the final tour by avoiding
    choices that would force long edges later. We score each candidate u by:
    score(u) = alpha * d(current, u) + beta * max_{v in unvisited\{u}} d(u, v)
    
    Here, the second term penalizes candidates that are far from some other unvisited node,
    which could lead to a long edge if we visit u now and then have to jump to a distant v later.
    However, pure minimax is often too conservative. So we blend it with nearest neighbor.
    
    Actually, a better structural change based on the directive:
    "selecting the next node that minimizes the distance to the farthest unvisited city"
    This is not quite right because we don't go to the farthest city next.
    
    Let's interpret "minimize the distance to the farthest unvisited city" as:
    For a candidate u, consider the set of remaining nodes R = unvisited \ {u}.
    The cost of picking u is related to how "isolated" u is from R.
    If u is far from all of R, it's a bad choice because it creates a long edge either entering or leaving u in the future?
    No, the edge entering u is d(current, u). The edge leaving u will be to some v in R.
    
    The directive says: "selecting the next node that minimizes the distance to the farthest unvisited city".
    This phrasing is slightly ambiguous. Does it mean:
    1. Minimize max_{v in R} d(u, v)? (Minimize the spread from u to others)
    2. Minimize d(u, farthest_unvisited)? (Where farthest is relative to current or destination?)
    
    Let's try a hybrid:
    We want to avoid long edges. A long edge is likely to occur if we pick a node u that is far from its nearest neighbor in the remaining set, OR if we pick a node that is far from the current node.
    
    Let's try the "Minimax" approach suggested:
    Score(u) = d(current, u) + min_{v in R} d(u, v) is the current 2-step lookahead.
    
    The directive suggests: "minimize the distance to the farthest unvisited city".
    Let's assume this means: For candidate u, look at the remaining nodes R.
    Find the node v in R that is farthest from u. Let max_d = max_{v in R} d(u, v).
    We want to minimize this? That would pick central nodes.
    
    But simply picking central nodes is not enough. We must also consider the step from current.
    
    Let's try:
    score(u) = d(current, u) + 0.5 * max_{v in R} d(u, v)
    
    Or perhaps the directive implies a different objective: Minimize the maximum edge in the *final* tour.
    This is the Min-Max TSP.
    
    Let's implement a heuristic that balances immediate cost and "regret" of not picking a central node.
    
    Actually, let's look at the "inspiration" 3: regret-based.
    
    Let's try a simple Minimax construction:
    1. Compute d_current = d(current, u) for all u.
    2. Compute d_unvisited = d(u, v) for all u, v in unvisited.
    3. For each u, compute max_dist_to_other = max_{v in R} d(u, v).
    4. Score = d_current + w * max_dist_to_other.
    
    Let's try w=0.1 or similar.
    
    Wait, if we minimize max_dist_to_other, we pick nodes that are close to ALL other unvisited nodes. This is like picking a centroid.
    This might lead to a "star" pattern which is bad for TSP (long closing edges).
    
    Maybe the directive means: "Minimize the distance to the farthest unvisited city *from the current node*"? No, that's just one value.
    
    Let's re-read: "selecting the next node that minimizes the distance to the farthest unvisited city".
    Maybe it means: For each candidate u, calculate the distance from u to the farthest unvisited city (excluding u). Let this be M(u).
    Choose u that minimizes M(u)?
    
    If we do that, we pick the most central node.
    
    Let's combine:
    score(u) = d(current, u) + lambda * M(u)
    where M(u) = max_{v in unvisited, v!=u} d(u, v).
    
    Let's try lambda = 0.1.
    """
    if len(unvisited_nodes) == 0:
        return -1

    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    if _check_timeout():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    # Vectorized implementation
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance matrix between unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute max distance from each u to any other v in unvisited (excluding itself)
    # Set diagonal to 0 so it doesn't affect max (distances are non-negative)
    d_unvisited_for_max = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_for_max, 0)
    
    max_dist_to_other = np.max(d_unvisited_for_max, axis=1)
    
    # Heuristic: Balance immediate distance and "centrality" (low max distance to others)
    # A node with low max_dist_to_other is "central" or "clumped".
    # A node with high max_dist_to_other is "peripheral" or "isolated".
    # We generally want to avoid isolated nodes until the end, but we also don't want to ignore proximity to current.
    
    # Let's try a weighted sum
    # If we pick a peripheral node now, we might create a long edge to enter or leave it.
    # But if we pick a central node, we might leave peripheral nodes for later, creating a long closing edge?
    
    # Standard TSP heuristics often prefer "nearest" to avoid long edges early.
    # Minimax TSP is different.
    
    # Let's try: score = d_current + 0.1 * max_dist_to_other
    score = d_current + 0.1 * max_dist_to_other
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
