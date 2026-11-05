import numpy as np
import math
import time

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels], dtype='U1')
    L = seqs.shape[1]
    
    if L == 0:
        return np.zeros((n, n)), labels

    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / L
    
    d = np.zeros((n, n))
    valid_mask = (p < 0.75) & (p > 0)
    zero_mask = (p == 0)
    inf_mask = (p >= 0.75)
    
    d[zero_mask] = 0.0
    d[valid_mask] = -0.75 * np.log(1.0 - 4.0 * p[valid_mask] / 3.0)
    d[inf_mask] = 10.0
    
    np.fill_diagonal(d, 0.0)
    return d, labels

def _neighbor_joining_optimized(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes))
    D[:n, :n] = dist
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Extract active sub-matrix
        d_active = D[np.ix_(active, active)]
        
        # Compute r: sum of distances from each node to all others in active
        r = d_active.sum(axis=1)
        
        # Compute Q matrix: (m-2) * d_active - r_i - r_j
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        
        # Set diagonal to inf to avoid self-joining
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        # Ensure i < j for consistency
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
            
        node_i = active[i_idx]
        node_j = active[j_idx]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # New tree string
        new_tree = f"({trees[node_i]},{trees[node_j]})"
        trees[new_id] = new_tree
        
        # Update distances from new_id to all other active nodes
        # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        
        other_indices = [idx for idx in range(m) if idx not in (i_idx, j_idx)]
        
        # Get the other node IDs
        other_nodes = [active[idx] for idx in other_indices]
        
        d_ij = D[node_i, node_j]
        
        # Calculate new distances
        d_i_x = D[node_i, other_nodes]
        d_j_x = D[node_j, other_nodes]
        
        new_dists = 0.5 * (d_i_x + d_j_x - d_ij)
        
        # Update D
        D[new_id, other_nodes] = new_dists
        D[other_nodes, new_id] = new_dists
        D[new_id, new_id] = 0.0
        
        # Remove i and j from active, add new_id
        active = [active[idx] for idx in other_indices] + [new_id]
        
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def reconstruct(sequences):
    start_time = time.time()
    deadline = 250.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if time.time() - start_time > deadline - 5:
        if n <= 2:
            return ",".join(labels) + ";"
        return "(" + ",".join(labels) + ");"
    
    tree_newick = _neighbor_joining_optimized(d, labels)
    
    for label in labels:
        if label not in tree_newick:
            return "(" + ",".join(labels) + ");"
            
    return tree_newick
