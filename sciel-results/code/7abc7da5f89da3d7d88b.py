"""Phylogenetic topology recovery using optimized Neighbor Joining with vectorized distance computation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_newick(tree, labels_map):
    """
    Recursively build Newick string from the internal tree structure.
    tree: dict or string node name
    labels_map: maps internal node ids to their structure
    """
    if isinstance(tree, str):
        return tree
    # If it's a tuple (left, right)
    if len(tree) == 2:
        left = _build_newick(tree[0], labels_map)
        right = _build_newick(tree[1], labels_map)
        return f"({left},{right})"
    return str(tree)

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 260.0  # 270s budget, leave 10s safety margin
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})();"
    if n_labels == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Convert sequences to integer matrix for fast comparison
    # A:0, C:1, G:2, T:3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    
    # Use vectorized conversion
    seq_chars = np.array([list(sequences[label]) for label in labels], dtype='U1')
    # Map characters to integers using vectorized approach
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for base, val in base_to_int.items():
        seq_mat[seq_chars == base] = val
    
    # Compute pairwise Hamming distances using vectorized operations
    # Use the fact that Hamming distance = sum of positions where sequences differ
    # For efficiency, compute using matrix operations
    # Method: for each position, compute which pairs differ
    # More efficient: use broadcasting but be careful with memory
    
    # Alternative: compute using the identity that d(i,j) = sum_k [seq[i,k] != seq[j,k]]
    # We can compute this as: for each pair (i,j), count mismatches
    # Vectorized approach: (n_labels, 1, L) vs (1, n_labels, L)
    # This creates (n_labels, n_labels, L) array
    
    # For n_labels up to 32 and L up to 2000: 32*32*2000 = 2,048,000 elements
    # That's about 2MB for int8, which is fine
    
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    # Normalize to p
    p = hamming.astype(np.float64) / L
    
    # Jukes-Cantor Correction
    # d = -0.75 * ln(1 - 4/3 * p)
    # Handle p >= 0.75 (which gives log of <= 0)
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    # Ensure symmetry and zero diagonal
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    
    # Initial nodes are the labels
    nodes = labels[:]
    D = dist_mat.copy()
    
    # Tree structure: map node name to (left_child, right_child)
    tree_struct = {label: label for label in labels}
    
    while len(nodes) > 3:
        if time.time() - start_time > time_limit:
            # Fallback: return a star tree for remaining nodes
            return f"({','.join(nodes)});"
        
        n = len(nodes)
        
        # Compute row sums r
        r = D.sum(axis=1)
        
        # Calculate Q matrix: Q_ij = (n-2)D_ij - r_i - r_j
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q pair (i, j) with i != j
        flat_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = flat_idx
        if i == j:
            # Should not happen due to inf diagonal
            continue
        
        node_i = nodes[i]
        node_j = nodes[j]
        
        # Create new node
        new_node = f"n{len(nodes)}_{i}_{j}"
        tree_struct[new_node] = (tree_struct[node_i], tree_struct[node_j])
        
        # Determine which indices to keep
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        kept_indices = np.where(keep_mask)[0]
        
        # New nodes list: kept nodes + new node
        new_nodes_list = [nodes[k] for k in kept_indices]
        new_nodes_list.append(new_node)
        
        # Construct new distance matrix D_new
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        # 1. Fill D_new for kept-kept distances
        D_keep_keep = D[np.ix_(kept_indices, kept_indices)]
        D_new[:-1, :-1] = D_keep_keep
        
        # 2. Fill D_new for kept-new_node distances
        # Formula: D_new(k, m) = 0.5 * (D(k, i) + D(k, j) - D(i, j))
        dist_i_kept = D[kept_indices, i]
        dist_j_kept = D[kept_indices, j]
        dist_ij = D[i, j]
        
        new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
        
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        
        # Update state
        nodes = new_nodes_list
        D = D_new
    
    # Final 3 nodes left: a, b, c
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
