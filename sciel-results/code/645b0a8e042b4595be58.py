"""Phylogenetic topology recovery using BionJ with JC correction."""
import numpy as np
import time

def _build_newick(tree, labels_map):
    if isinstance(tree, str):
        return tree
    if len(tree) == 2:
        return f"({_build_newick(tree[0], labels_map)},{_build_newick(tree[1], labels_map)})"
    return str(tree)

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 250.0
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})();"

    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq_mat[i] = np.array([base_to_int[b] for b in sequences[label]], dtype=np.int8)
        
    # Compute Hamming distances
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    # BionJ Step 1: Raw distances
    # We use Hamming distances as raw d
    d_raw = hamming.astype(np.float64)
    
    # BionJ Step 2: Compute BionJ distances
    # d'_ij = 0.5 * sum_{k != i, j} (d_ik + d_jk - d_ij) / (n-3)
    # Note: sum is over n-2 terms.
    # Let S_i = sum_k d_ik.
    # sum_{k != i, j} (d_ik + d_jk - d_ij) = (S_i - d_ij) + (S_j - d_ij) - (n-2)d_ij
    # = S_i + S_j - 2*d_ij - (n-2)d_ij
    # = S_i + S_j - (n)d_ij
    
    S = d_raw.sum(axis=1)
    # d'_ij = 0.5 * (S_i + S_j - n * d_ij) / (n-3)
    # We need to do this for all i, j.
    
    # S_i is (n,), S_j is (n,).
    # S_i + S_j is (n, n) via broadcasting.
    # d_ij is (n, n).
    
    if n_labels > 3:
        d_bionj_raw = 0.5 * (S[:, None] + S[None, :] - n_labels * d_raw) / (n_labels - 3)
    else:
        d_bionj_raw = d_raw
        
    # BionJ Step 3: JC Correction
    # The BionJ distance is a substitution rate.
    # p = d_bionj_raw / L ?
    # The raw Hamming distance is a count.
    # The BionJ formula uses distances. If we use Hamming counts, the result is in counts.
    # We need to convert to probability p.
    # p = d_bionj_raw / L
    
    p = d_bionj_raw / L
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    # Ensure symmetry and zero diagonal
    dist_mat = np.triu(dist_mat)
    dist_mat = dist_mat + dist_mat.T
    np.fill_diagonal(dist_mat, 0.0)
    
    # Neighbor Joining
    nodes = labels[:]
    D = dist_mat[:n_labels, :n_labels].copy()
    
    tree_struct = {label: label for label in labels}
    next_internal_id = 0
    
    while len(nodes) > 3:
        n = len(nodes)
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_val = np.inf
        min_i, min_j = -1, -1
        flat_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = flat_idx
        
        node_i = nodes[i]
        node_j = nodes[j]
        
        new_node = f"n{next_internal_id}"
        next_internal_id += 1
        tree_struct[new_node] = (tree_struct[node_i], tree_struct[node_j])
        
        new_nodes_list = []
        for k, node in enumerate(nodes):
            if node == node_i or node == node_j:
                continue
            new_nodes_list.append(node)
        new_nodes_list.append(new_node)
        
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        kept_indices = []
        for node in new_nodes_list[:-1]:
            idx = nodes.index(node)
            kept_indices.append(idx)
            
        D_keep_keep = D[np.ix_(kept_indices, kept_indices)]
        D_new[:-1, :-1] = D_keep_keep
        
        dist_i_kept = D[kept_indices, i]
        dist_j_kept = D[kept_indices, j]
        dist_ij = D[i, j]
        
        new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
        
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        
        nodes = new_nodes_list
        D = D_new
        
    if len(nodes) == 3:
        a, b, c = nodes
        tree_struct['final_root'] = (tree_struct[a], (tree_struct[b], tree_struct[c]))
        final_node = 'final_root'
    elif len(nodes) == 2:
        a, b = nodes
        tree_struct['final_root'] = (tree_struct[a], tree_struct[b])
        final_node = 'final_root'
    else:
        final_node = nodes[0]
        
    newick = _build_newick(tree_struct[final_node], tree_struct)
    return newick + ";"

# EVOLVE-BLOCK-END
