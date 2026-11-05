import numpy as np
import math
import time

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels])
    
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / seqs.shape[1]
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    D = dist.copy()
    node_ids = list(range(n))
    trees = [labels[i] for i in range(n)]
    
    while len(node_ids) > 3:
        m = len(node_ids)
        
        # Calculate r (sum of distances to all others)
        r = D.sum(axis=1)
        
        # Calculate Q matrix
        Q = (m - 2) * D - r[:, None] - r[None, :]
        
        # Mask diagonal
        np.fill_diagonal(Q, float('inf'))
        
        # Find min Q
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        
        if i == j:
            break
            
        # Calculate new distances
        keep_indices = [k for k in range(m) if k not in (i, j)]
        
        D_keep = D[np.ix_(keep_indices, keep_indices)]
        
        d_i_keep = D[i, keep_indices]
        d_j_keep = D[j, keep_indices]
        d_ij = D[i, j]
        
        d_new_keep = 0.5 * (d_i_keep + d_j_keep - d_ij)
        
        # Construct new D
        D_new = np.zeros((m-1, m-1))
        D_new[:m-2, :m-2] = D_keep
        D_new[:m-2, m-1] = d_new_keep
        D_new[m-1, :m-2] = d_new_keep
        D_new[m-1, m-1] = 0.0
        
        D = D_new
        
        # Update node_ids and trees
        new_node_ids = [node_ids[k] for k in keep_indices]
        new_node_ids.append(len(node_ids)) # Unique ID for new node
        node_ids = new_node_ids
        
        new_trees = [trees[k] for k in keep_indices]
        new_trees.append(f"({trees[i]},{trees[j]})")
        trees = new_trees
        
    # Final 3 nodes
    a, b, c = node_ids
    # trees are aligned with node_ids
    tree_a = trees[0]
    tree_b = trees[1]
    tree_c = trees[2]
    
    return f"({tree_a},({tree_b},{tree_c}));"

def reconstruct(sequences):
    start_time = time.time()
    
    d, labels = _jc_distance_matrix(sequences)
    tree_newick = _neighbor_joining(d, labels)
    
    return tree_newick
