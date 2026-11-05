"""Distance-based phylogenetic tree reconstruction using Neighbor-Joining with Jukes-Cantor correction."""
import numpy as np
import math


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0..t(n-1)) to DNA strings over ACGT
        
    Returns:
        Newick string representing the unrooted tree topology
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Convert sequences to numpy arrays for faster distance computation
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    
    # Pre-convert sequences to integer arrays for faster comparison
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_seqs = np.array([[base_to_int[c] for c in s] for s in seq_list], dtype=np.int8)
    
    # Compute pairwise Jukes-Cantor corrected distances using vectorized operations
    # D[a,b] = -0.75 * ln(1 - 4p/3) where p = proportion of differing sites
    # Compute all pairwise Hamming distances at once
    # For each pair (i,j), count positions where int_seqs[i] != int_seqs[j]
    # We can use broadcasting: diff_matrix[i,j] = sum over sites of (int_seqs[i,s] != int_seqs[j,s])
    
    # Efficient approach: for each site, compute which pairs differ
    # Or use the identity: sum_{s} [x_s != y_s] = sum_s x_s + sum_s y_s - 2*sum_s [x_s == y_s]
    # Actually simpler: use np.apply_along_axis or manual loop over sites with vectorized pair ops
    
    # Most efficient: compute column sums and use matrix operations
    # For binary encoding trick or direct comparison
    
    # Direct approach: compute all pairwise differences
    # diff[i,j] = sum over k of (int_seqs[i,k] != int_seqs[j,k])
    # This is O(n^2 * L) but with numpy vectorization per site
    
    # Better: use the fact that for 4 bases, we can use one-hot or direct comparison
    # Let's just do it site by site with vectorized pair computation
    
    diff_counts = np.zeros((n, n), dtype=np.int32)
    for s in range(L):
        col = int_seqs[:, s]  # shape (n,)
        # For each pair (i,j), count if col[i] != col[j]
        # Create difference matrix
        diff_mat = (col[:, None] != col[None, :]).astype(np.int32)
        diff_counts += diff_mat
    
    # Convert to Jukes-Cantor distances
    p = diff_counts / L
    np.fill_diagonal(p, 0.0)
    
    # Apply JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 case
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Clamp values where p >= 0.75 (log of negative or zero)
    dist[~np.isfinite(dist)] = 3.0
    np.fill_diagonal(dist, 0.0)
    
    # Neighbor-Joining algorithm with vectorized operations
    # Maintain a distance matrix over all potential nodes (up to 2n-2 nodes)
    # active: list of current node indices
    # D_full: (2n-2) x (2n-2) distance matrix
    # tree_str: dict mapping node_id -> Newick string
    
    max_nodes = 2 * n - 2
    D_full = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D_full[:n, :n] = dist
    
    tree_str = {i: labels[i] for i in range(n)}
    active = list(range(n))
    next_node_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Extract current distance submatrix
        idx = np.array(active)
        D_sub = D_full[idx[:, None], idx[None, :]]
        
        # Compute row sums
        r = D_sub.sum(axis=1)
        
        # Compute Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair with minimum Q
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx[0], min_idx[1]
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Create a new internal node joining a and b
        new_node = next_node_id
        next_node_id += 1
        
        # Compute branch lengths
        delta_a = 0.5 * (D_sub[i_idx, j_idx] + (r[i_idx] - r[j_idx]) / (m - 2))
        delta_b = D_sub[i_idx, j_idx] - delta_a
        
        # Ensure non-negative
        delta_a = max(0.0, delta_a)
        delta_b = max(0.0, delta_b)
        
        # Build the subtree string
        tree_str[new_node] = f"({tree_str[a]}:{delta_a:.4f},{tree_str[b]}:{delta_b:.4f})"
        
        # Update distances from new_node to all remaining active nodes
        remaining_idx = [x for x in range(m) if x not in (i_idx, j_idx)]
        for x_idx in remaining_idx:
            x = active[x_idx]
            # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            new_d = 0.5 * (D_sub[i_idx, x_idx] + D_sub[j_idx, x_idx] - D_sub[i_idx, j_idx])
            D_full[new_node, x] = new_d
            D_full[x, new_node] = new_d
        
        # Remove a and b from active, add new_node
        active = [active[x] for x in remaining_idx] + [new_node]
    
    # Final step: join the remaining 3 nodes into an unrooted tree
    a, b, c = active
    
    dab = D_full[a, b]
    dac = D_full[a, c]
    dbc = D_full[b, c]
    
    # Branch lengths for 3-taxon unrooted tree
    len_a = max(0.0, (dab + dac - dbc) / 2.0)
    len_b = max(0.0, (dab + dbc - dac) / 2.0)
    len_c = max(0.0, (dac + dbc - dab) / 2.0)
    
    # Build final Newick
    result = f"({tree_str[a]}:{len_a:.4f},{tree_str[b]}:{len_b:.4f},{tree_str[c]}:{len_c:.4f})"
    
    return result
