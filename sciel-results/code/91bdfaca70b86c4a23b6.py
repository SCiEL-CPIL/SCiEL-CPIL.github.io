"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 3-step lookahead heuristic.
    
    For each candidate node u in unvisited_nodes, we estimate the cost as:
    dist(current, u) + dist(u, v) + dist(v, w)
    
    where v is the nearest unvisited node to u (excluding u),
    and w is the nearest unvisited node to v (excluding u and v).
    
    This deeper lookahead helps avoid local minima more effectively than 2-step.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        # 2-step lookahead is sufficient for 2 nodes
        u1 = unvisited_nodes[0]
        u2 = unvisited_nodes[1]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2]
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1]
        return int(unvisited_nodes[0] if cost1 < cost2 else unvisited_nodes[1])

    # Get distances from current_node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Iterate through each unvisited node to evaluate the 3-step cost
    for i, u in enumerate(unvisited_nodes):
        # Cost to reach u
        cost_to_u = dist_from_current[i]
        
        # Find the nearest unvisited node to u (v), excluding u
        dists_from_u = distance_matrix[u, unvisited_nodes]
        
        # Mask out u
        mask_u = np.ones(n_unvisited, dtype=bool)
        mask_u[i] = False
        
        # Find v (index in unvisited_nodes)
        # We need the index of the min in dists_from_u where mask_u is True
        # Set dists_from_u[i] to inf
        dists_from_u_masked = dists_from_u.copy()
        dists_from_u_masked[i] = np.inf
        
        if n_unvisited > 2:
            v_idx = np.argmin(dists_from_u_masked)
            v = unvisited_nodes[v_idx]
            
            # Find the nearest unvisited node to v (w), excluding u and v
            dists_from_v = distance_matrix[v, unvisited_nodes]
            
            # Mask out u and v
            mask_v = np.ones(n_unvisited, dtype=bool)
            mask_v[i] = False
            mask_v[v_idx] = False
            
            dists_from_v_masked = dists_from_v.copy()
            dists_from_v_masked[i] = np.inf
            dists_from_v_masked[v_idx] = np.inf
            
            w_idx = np.argmin(dists_from_v_masked)
            w = unvisited_nodes[w_idx]
            
            # Cost: c->u + u->v + v->w
            # Note: We don't add w->dest here as that would be 4-step
            # 3-step means we look 3 nodes ahead? Or 3 edges?
            # Usually "k-step lookahead" means looking k steps ahead in the path.
            # Path: current -> u -> v -> w.
            # Cost is sum of these 3 edges.
            total_cost = cost_to_u + distance_matrix[u, v] + distance_matrix[v, w]
        else:
            # n_unvisited == 2 handled above, so this case is n_unvisited > 2 always here
            # But just in case, fallback to 2-step logic
            v_idx = np.argmin(dists_from_u_masked)
            v = unvisited_nodes[v_idx]
            total_cost = cost_to_u + distance_matrix[u, v]

        if total_cost < best_score:
            best_score = total_cost
            best_node = u

    return int(best_node)
# EVOLVE-BLOCK-END
