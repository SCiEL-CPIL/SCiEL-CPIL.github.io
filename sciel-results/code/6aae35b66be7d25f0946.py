"""Rewritten program: TSP constructive heuristic using dynamic potential field."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Dynamic Potential Field.
    
    The next node is selected by minimizing a cost function composed of:
    1. Attractive force: Distance from current node.
    2. Repulsive force: Sum of inverse distances to all other unvisited nodes.
       This term encourages visiting nodes that are 'isolated' from the remaining
       set, promoting spatial coverage and preventing clustering.
    3. Destination pull: Distance to the destination node, weighted to increase
       as the number of unvisited nodes decreases.
    
    cost(u) = d(current, u) + alpha * sum_{v != u} 1/d(u, v) + beta * d(u, destination)
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        # For 2 nodes, pick the one that minimizes the total path
        # current -> u -> v -> destination
        u0 = int(unvisited_nodes[0])
        u1 = int(unvisited_nodes[1])
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        return u0 if cost0 <= cost1 else u1
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    
    # 3. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Compute Repulsive Potential
    # Set diagonal to a large number to avoid division by zero, 
    # effectively ignoring self-interaction.
    # We want sum_{v != u} 1 / d(u, v)
    # Using np.where to handle diagonal safely
    inv_d_sub = np.where(d_sub > 1e-9, 1.0 / d_sub, 0.0)
    # The diagonal is 0 in d_sub (distance to self), so 1/0 is handled as 0.
    # Sum over axis=1 gives the repulsive potential for each node u.
    repulsive_potential = np.sum(inv_d_sub, axis=1)
    
    # 5. Adaptive Weighting
    # The repulsive force should be strong when many nodes remain to ensure coverage.
    # As nodes are visited, the "fluid" density drops, and we can rely more on distance.
    # However, a constant or slowly decaying alpha is often robust.
    # Let's try a scaling that keeps the magnitude of repulsion comparable to distances.
    # Typical distance is O(1). Repulsion is sum of ~n terms of O(1/d).
    # If nodes are uniform in unit square, average d ~ 0.5, 1/d ~ 2. Sum ~ 2n.
    # So alpha should be small, maybe 1/n or similar to balance.
    
    # Let's use a simple heuristic:
    # alpha = 0.5 * (n / 50.0)  # Scale down as n decreases? Or keep constant?
    # Actually, if we want to prevent clustering, we want high repulsion.
    # Let's try a fixed alpha relative to the scale of distances.
    # A common choice is to normalize the repulsion.
    
    # Let's experiment with:
    # cost = d_curr + alpha * repulsive_potential + beta * d_dest
    
    # If alpha is too large, we pick the most isolated point (worst for TSP path continuity).
    # If alpha is too small, it's just Nearest Neighbor.
    # A good balance might be alpha ~ 0.1 to 0.5.
    
    # Let's normalize repulsive potential by the number of nodes to get an average repulsion?
    # Or just use the sum.
    
    # Let's try:
    alpha = 0.3 * (n / 50.0)  # Weaker repulsion when fewer nodes left
    beta = 0.2 * (1.0 / (1.0 + n / 10.0)) # Stronger destination pull when few nodes left
    
    # 6. Calculate Scores
    scores = d_curr + alpha * repulsive_potential + beta * d_dest
    
    # 7. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
