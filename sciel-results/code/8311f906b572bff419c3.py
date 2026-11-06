"""Optimized Neighbor-Joining with vectorized distance computation and inverse matrix method."""
import numpy as np


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Convert sequences to integer matrix: A=0, C=1, G=2, T=3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Create matrix of shape (n, length)
    seq_matrix = np.zeros((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        for j, ch in enumerate(seq):
            seq_matrix[i, j] = base_to_int[ch]
    
    # Compute pairwise Hamming distances using broadcasting
    # diff_matrix[i, j] = number of positions where seq i and seq j differ
    diff_matrix = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2).astype(np.float64)
    
    # Compute p = diff / length
    p = diff_matrix / length
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (saturation) by capping
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining_vectorized(d):
    """
    Vectorized Neighbor-Joining algorithm using pre-computed inverse matrix.
    Returns a Newick string.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    # Work with a copy of the distance matrix
    D = d.astype(np.float64).copy()
    
    # Track the current tree structure for each node
    # tree[i] is the Newick substring for node i
    tree = [f"t{i}" for i in range(n)]
    
    # Active nodes - we'll track which nodes are still in the tree
    active = set(range(n))
    
    # We'll maintain a mapping from active node indices to their positions in the working array
    # For efficiency, we'll work with a dense representation and mask out inactive nodes
    
    # Pre-allocate arrays for up to 2n nodes (n leaves + n-2 internal)
    max_nodes = 2 * n
    D_full = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D_full[:n, :n] = D
    
    tree_full = [f"t{i}" for i in range(n)]
    active_list = list(range(n))
    
    while len(active_list) > 2:
        m = len(active_list)
        
        # Extract sub-matrix of active nodes
        idx = np.array(active_list)
        D_sub = D_full[np.ix_(idx, idx)]
        
        # Compute row sums (excluding diagonal)
        r = D_sub.sum(axis=1)
        
        # Compute Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        # Vectorized computation
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        
        # Set diagonal to infinity to avoid self-pairs
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair with minimum Q
        min_val = Q.min()
        # Get the first occurrence of the minimum
        min_pos = np.argmin(Q)
        i_idx = min_pos // m
        j_idx = min_pos % m
        
        # Ensure i_idx < j_idx
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        a = active_list[i_idx]
        b = active_list[j_idx]
        
        # Compute branch lengths
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        len_a = 0.5 * D_sub[i_idx, j_idx] + 0.5 * delta
        len_b = 0.5 * D_sub[i_idx, j_idx] - 0.5 * delta
        
        # Ensure non-negative
        len_a = max(len_a, 0.0)
        len_b = max(len_b, 0.0)
        
        # Create new node
        new_label = f"({tree_full[a]}:{len_a:.4f},{tree_full[b]}:{len_b:.4f})"
        
        # Find next available node index
        new_node = max(active_list) + 1
        
        # Update distances from new node to all other active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        for pos, x in enumerate(active_list):
            if x != a and x != b:
                D_full[new_node, x] = 0.5 * (D_full[a, x] + D_full[b, x] - D_full[a, b])
                D_full[x, new_node] = D_full[new_node, x]
        
        # Remove a and b from active, add new_node
        active_list = [x for x in active_list if x not in (a, b)]
        active_list.append(new_node)
        
        # Store the new tree label
        tree_full.append(new_label)
    
    # Combine the last two nodes
    if len(active_list) == 2:
        i, j = active_list
        d_ij = D_full[i, j]
        half = max(d_ij / 2.0, 0.0)
        newick = f"({tree_full[i]}:{half:.4f},{tree_full[j]}:{half:.4f})"
    else:
        newick = tree_full[active_list[0]]
    
    return newick + ";"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0, t1, ...) to DNA strings
        
    Returns:
        Newick string
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Run vectorized Neighbor-Joining
    newick = _neighbor_joining_vectorized(D)
    
    return newick
