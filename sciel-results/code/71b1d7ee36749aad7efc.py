"""Optimized Neighbor-Joining with fully vectorized operations."""
import numpy as np


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Convert sequences to integer matrix: A=0, C=1, G=2, T=3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.array([[base_map[ch] for ch in seq] for seq in seq_list], dtype=np.int8)
    
    # Compute pairwise Hamming distances via broadcasting
    diff_matrix = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2).astype(np.float64)
    
    # Jukes-Cantor correction
    p = diff_matrix / length
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining(d):
    """
    Vectorized Neighbor-Joining algorithm on a distance matrix d of shape (n, n).
    Returns a Newick string.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    # D is the full distance matrix; we'll track active indices
    D = d.copy()
    tree = [f"t{i}" for i in range(n)]
    active = list(range(n))
    
    while len(active) > 2:
        n_active = len(active)
        
        # Build the sub-matrix for active nodes
        idx = np.array(active)
        sub_D = D[np.ix_(idx, idx)]
        
        # Row sums (excluding diagonal)
        r = sub_D.sum(axis=1)
        
        # Q matrix: Q[i,j] = (n_active - 2) * D[i,j] - r[i] - r[j]
        Q = (n_active - 2) * sub_D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find the minimum Q entry (off-diagonal)
        i_local, j_local = np.unravel_index(np.argmin(Q), Q.shape)
        if i_local > j_local:
            i_local, j_local = j_local, i_local
        
        a = active[i_local]
        b = active[j_local]
        
        # Branch lengths
        delta = (r[i_local] - r[j_local]) / (2.0 * (n_active - 2))
        len_a = max(0.0, 0.5 * sub_D[i_local, j_local] + 0.5 * delta)
        len_b = max(0.0, 0.5 * sub_D[i_local, j_local] - 0.5 * delta)
        
        # Create new node label
        new_label = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Compute distances from new node to remaining active nodes
        remaining = [x for k, x in enumerate(active) if k not in (i_local, j_local)]
        
        new_node = max(active) + 1
        
        # Extend D
        old_n = D.shape[0]
        new_D = np.zeros((old_n + 1, old_n + 1))
        new_D[:old_n, :old_n] = D
        
        for x in remaining:
            new_D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            new_D[x, new_node] = new_D[new_node, x]
        
        D = new_D
        tree.append(new_label)
        active = remaining + [new_node]
    
    # Combine the last two nodes
    if len(active) == 2:
        i, j = active
        d_ij = D[i, j]
        half = d_ij / 2.0
        newick = f"({tree[i]}:{half:.4f},{tree[j]}:{half:.4f})"
    else:
        newick = tree[active[0]]
    
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
    
    # Run Neighbor-Joining
    newick = _neighbor_joining(D)
    
    return newick
