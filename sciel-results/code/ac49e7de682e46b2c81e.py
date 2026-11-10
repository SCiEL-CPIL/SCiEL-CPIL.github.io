"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances using NumPy for speed.'''
import numpy as np
import time
import math


def _jc_distance_matrix(sequences):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = []
    for label in labels:
        seq = sequences[label]
        arr = np.array([code_map[c] for c in seq], dtype=np.int8)
        seq_arrays.append(arr)
    
    matrix = np.vstack(seq_arrays)
    diff = (matrix[:, np.newaxis, :] != matrix[np.newaxis, :, :])
    hamming = diff.sum(axis=2)
    
    p = hamming / matrix.shape[1]
    
    valid = p < 0.75
    dist_matrix = np.full_like(p, 3.0, dtype=np.float64)
    
    arg = 1.0 - (4.0 / 3.0) * p[valid]
    dist_matrix[valid] = -0.75 * np.log(arg)
    
    np.fill_diagonal(dist_matrix, 0.0)
    
    return labels, dist_matrix


def reconstruct(sequences):
    start_time = time.time()
    time_limit = 260.0
    
    labels, dist_matrix = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
        
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist_matrix
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    new_node_id = n
    
    while len(nodes) > 3:
        if time.time() - start_time > time_limit:
            # Fallback: return a star tree if we time out (unlikely)
            # But we need to return a valid Newick.
            # If we break here, we don't have a complete tree.
            # We'll just let it run, as it should be fast.
            pass
            
        m = len(nodes)
        D_sub = D[np.ix_(nodes, nodes)]
        r = np.sum(D_sub, axis=1)
        Q = (m - 2) * D_sub - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        i, j = nodes[i_idx], nodes[j_idx]
        
        k = new_node_id
        new_node_id += 1
        
        trees[k] = f"({trees[i]},{trees[j]})"
        
        other_indices = [x for x in nodes if x not in (i, j)]
        
        if other_indices:
            vals = 0.5 * (D[i, other_indices] + D[j, other_indices] - D[i, j])
            D[k, other_indices] = vals
            D[other_indices, k] = vals
            D[k, k] = 0.0
            
        nodes.remove(i)
        nodes.remove(j)
        nodes.append(k)
        
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = D[a, b]
        d_ac = D[a, c]
        d_bc = D[b, c]
        
        if d_ab <= d_ac and d_ab <= d_bc:
            i, j, k = a, b, c
        elif d_ac <= d_ab and d_ac <= d_bc:
            i, j, k = a, c, b
        else:
            i, j, k = b, c, a
            
        new_k1 = new_node_id
        new_node_id += 1
        trees[new_k1] = f"({trees[i]},{trees[j]})"
        
        new_k2 = new_node_id
        new_node_id += 1
        trees[new_k2] = f"({trees[new_k1]},{trees[k]})"
        
        return trees[new_k2] + ";"
        
    return ""
# EVOLVE-BLOCK-END
