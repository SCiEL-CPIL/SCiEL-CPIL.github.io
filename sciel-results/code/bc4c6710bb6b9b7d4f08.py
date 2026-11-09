# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time, _DEADLINE
    
    if time.time() - _start_time > _DEADLINE:
        if len(unvisited_nodes) == 0:
            return -1
        d_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(d_current)])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Phase-aware: pure NN early, blend in destination proximity late
    # Total nodes = 50 (from task spec). unvisited count tells us progress.
    # When unvisited is large (>40), use pure NN.
    # When unvisited is small (<=10), blend in distance to destination.
    # The blend weight increases as we get closer to the end.
    
    d_current = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Determine blend weight: 0.0 for first 40 picks, ramps up for last 10
    # n_unvisited goes from 49 down to 1.
    # We want blend = 0 when n_unvisited > 40
    # blend ramps from 0 to 1 as n_unvisited goes from 40 to 1
    if n_unvisited > 40:
        blend = 0.0
    else:
        # n_unvisited in [1, 40]: map to [0, 1]
        # at n_unvisited=40, blend=0; at n_unvisited=1, blend=1
        blend = (40.0 - n_unvisited) / 39.0
    
    if blend == 0.0:
        best_idx = np.argmin(d_current)
    else:
        # Weighted score: lower is better
        # score = (1-blend) * d_current + blend * d_dest
        # Normalize both to [0,1] for fair blending
        d_curr_min = np.min(d_current)
        d_curr_max = np.max(d_current)
        d_dest_min = np.min(d_dest)
        d_dest_max = np.max(d_dest)
        
        if d_curr_max > d_curr_min:
            d_curr_norm = (d_current - d_curr_min) / (d_curr_max - d_curr_min)
        else:
            d_curr_norm = np.zeros_like(d_current)
            
        if d_dest_max > d_dest_min:
            d_dest_norm = (d_dest - d_dest_min) / (d_dest_max - d_dest_min)
        else:
            d_dest_norm = np.zeros_like(d_dest)
        
        score = (1.0 - blend) * d_curr_norm + blend * d_dest_norm
        best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
