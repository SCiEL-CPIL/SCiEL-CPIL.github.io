# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step Lookahead heuristic.
    This approach estimates the cost of moving to a candidate node and then
    moving to its nearest neighbor, avoiding short-sighted choices that lead
    to high-cost jumps later.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Get indices of unvisited nodes
    # We need to evaluate each candidate u in unvisited_nodes
    
    best_score = float('inf')
    best_node = unvisited_nodes[0]
    
    # Pre-fetch distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    for i, u in enumerate(unvisited_nodes):
        # Cost to go from current to u
        cost_to_u = dists_from_current[i]
        
        # If u is the last one, we just add the distance to destination?
        # The problem says "closed tour", so the final step goes back to destination_node.
        # However, select_next_node is called until unvisited is empty.
        # If len(unvisited) == 2, we pick one, then the last one is forced.
        # So for the second to last node, the "next" step is to the last node, then to destination.
        # Actually, the heuristic should just look at the immediate next step in the chain.
        
        # Find the nearest neighbor of u among the other unvisited nodes
        if len(unvisited_nodes) > 2:
            # Create a mask to exclude u
            mask = np.ones(len(unvisited_nodes), dtype=bool)
            mask[i] = False
            other_unvisited = unvisited_nodes[mask]
            
            # Distances from u to other unvisited
            dists_from_u = distance_matrix[u, other_unvisited]
            min_dist_from_u = np.min(dists_from_u)
        else:
            # If only 2 nodes left, the "next" node is the other one.
            # The cost is dist(current, u) + dist(u, other).
            # But we don't know which one is picked yet.
            # If we pick u, the next is the other one.
            # If we pick the other one, the next is u.
            # The lookahead for u would be dist(u, other).
            # The lookahead for other would be dist(other, u).
            # They are the same.
            # So the score is d1 + d2.
            # Let's just calculate it generally.
            other_unvisited = unvisited_nodes[[j for j in range(len(unvisited_nodes)) if j != i]]
            if len(other_unvisited) == 0:
                min_dist_from_u = 0.0
            else:
                dists_from_u = distance_matrix[u, other_unvisited]
                min_dist_from_u = np.min(dists_from_u)
        
        # Total lookahead score
        # We want to minimize the path.
        # score = dist(current, u) + min_dist_from_u
        score = cost_to_u + min_dist_from_u
        
        # Optional: Add a small bias towards destination if we are near the end?
        # No, keep it simple first.
        
        if score < best_score:
            best_score = score
            best_node = u
            
    return int(best_node)
# EVOLVE-BLOCK-END
