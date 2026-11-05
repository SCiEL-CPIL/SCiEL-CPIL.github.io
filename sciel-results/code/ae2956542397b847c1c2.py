"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Deadline guard: fall back to nearest-neighbor if time is running out
    if time.time() > _DEADLINE:
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    dm = distance_matrix
    best_score = np.inf
    best_node = int(unv[0])
    
    # Precompute distance from destination to all nodes for return cost estimation
    dest_dists = dm[destination_node, :]
    
    for c in unv:
        d1 = dm[current_node, c]
        # Pruning: if immediate cost is already worse than best total, skip
        if d1 >= best_score:
            continue
        
        # Lookahead depth 2
        remaining1 = unv[unv != c]
        dists_c = dm[c, remaining1]
        idx_x = np.argmin(dists_c)
        x = remaining1[idx_x]
        d2 = dists_c[idx_x]
        
        score = d1 + d2
        
        # Lookahead depth 3
        if n > 2:
            remaining2 = remaining1[remaining1 != x]
            dists_x = dm[x, remaining2]
            idx_y = np.argmin(dists_x)
            y = remaining2[idx_y]
            d3 = dists_x[idx_y]
            
            score += d3
            
            # Lookahead depth 4 and return penalty
            if n > 3:
                remaining3 = remaining2[remaining2 != y]
                dists_y = dm[y, remaining3]
                idx_z = np.argmin(dists_y)
                z = remaining3[idx_z]
                d4 = dists_y[idx_z]
                
                score += d4
                
                # Estimate cost to return to destination from z
                # We weight this heavily to avoid nodes that are far from the destination
                score += 1.0 * dm[z, destination_node]
        
        if score < best_score:
            best_score = score
            best_node = int(c)
            
    return best_node
# EVOLVE-BLOCK-END
