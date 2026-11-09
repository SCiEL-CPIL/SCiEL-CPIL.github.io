"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987) with NumPy acceleration.'''
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 270.0  # 270s wall clock limit

def _jc_distance_matrix(sequences):
    """
    Compute JC distance matrix using NumPy.
    sequences: dict {label: string}
    Returns: (labels, distance_matrix)
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 0:
        return labels, np.array([])
    L = len(sequences[labels[0]])
    
    if L == 0:
        return labels, np.zeros((n, n))
        
    # Map A,C,G,T to 0,1,2,3
    map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    seq_matrix = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, char in enumerate(seq):
            seq_matrix[i, j] = map_dict.get(char, 0)
            
    # Compute Hamming distances
    diff = seq_matrix[:, None, :] != seq_matrix[None, :, :]
    hamming = np.sum(diff, axis=2)
    
    p = hamming / L
    
    # JC correction
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0 / 3.0) * p_safe)
    
    # Handle saturation
    d[p >= 0.75] = 3.0
    
    return labels, d

def reconstruct(sequences):
    start_time = time.time()
    labels, dist = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
        
    # NJ Implementation
    active = list(range(n))
    max_nodes = 2 * n - 1
    work_dist = np.zeros((max_nodes, max_nodes))
    work_dist[:n, :n] = dist
    
    tree_strs = [labels[i] for i in range(n)]
    next_node_id = n
    
    while len(active) > 3:
        if time.time() - _START_TIME > _DEADLINE - 5:
            # Return best effort (current tree)
            break
            
        k = len(active)
        sub_dist = work_dist[np.ix_(active, active)]
        row_sums = sub_dist.sum(axis=1)
        
        Q = (k - 2) * sub_dist - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_idx, Q.shape)
        
        node_i = active[i_idx]
        node_j = active[j_idx]
        
        new_node = next_node_id
        next_node_id += 1
        
        other_indices = [idx for idx in range(len(active)) if idx != i_idx and idx != j_idx]
        other_nodes = [active[idx] for idx in other_indices]
        
        d_ij = work_dist[node_i, node_j]
        
        if other_nodes:
            d_i_o = work_dist[node_i, other_nodes]
            d_j_o = work_dist[node_j, other_nodes]
            d_new_o = 0.5 * (d_i_o + d_j_o - d_ij)
            
            work_dist[new_node, other_nodes] = d_new_o
            work_dist[other_nodes, new_node] = d_new_o
            
        work_dist[new_node, new_node] = 0.0
        
        tree_strs.append(f"({tree_strs[node_i]},{tree_strs[node_j]})")
        
        active = [x for x in active if x != node_i and x != node_j]
        active.append(new_node)
        
    final_nodes = active
    final_strs = [tree_strs[x] for x in final_nodes]
    
    return "(" + ",".join(final_strs) + ");"
# EVOLVE-BLOCK-END
