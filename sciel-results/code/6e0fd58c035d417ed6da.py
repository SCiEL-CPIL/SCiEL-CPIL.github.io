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
    base_to_int = np.array([0, 1, 2, 3], dtype=np.int8)
    # Map: A=0, C=1, G=2, T=3
    char_map = np.array([ord(c) for c in 'ACGT'], dtype=np.int32)
    int_seqs = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seq_list):
        # Convert string to array of ordinals, then map to 0-3
        ords = np.frombuffer(s.encode(), dtype=np.int8)
        # Build lookup: A(65)->0, C(67)->1, G(71)->2, T(84)->3
        # Use a simple approach with np.where or a lookup table
        lookup = np.zeros(256, dtype=np.int8)
        lookup[ord('A')] = 0
        lookup[ord('C')] = 1
        lookup[ord('G')] = 2
        lookup[ord('T')] = 3
        int_seqs[i] = lookup[ords]
    
    # Compute pairwise Jukes-Cantor corrected distances using vectorized operations
    # Count differences for all pairs at once
    # diff_matrix[i,j] = number of positions where seq i and seq j differ
    # Use broadcasting: int_seqs[:, np.newaxis, :] != int_seqs[np.newaxis, :, :]
    # This creates an (n, n, L) boolean array - might be memory intensive for large L
    # For n=32, L=2000: 32*32*2000 = 2M booleans = 2MB, fine
    
    # Alternative: compute per-position differences
    # diff[i,j,k] = 1 if int_seqs[i,k] != int_seqs[j,k]
    diff_3d = (int_seqs[:, np.newaxis, :] != int_seqs[np.newaxis, :, :])
    diff_counts = diff_3d.sum(axis=2)  # (n, n) matrix of difference counts
    
    # Compute JC distances
    p = diff_counts.astype(np.float64) / L
    # Clip to avoid log of negative or zero
    p = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    
    # Neighbor-Joining algorithm with vectorized operations
    # We maintain:
    #   active: list of node indices currently in the NJ process
    #   D: distance matrix (full n x (2n-2) sized, but we only use active portion)
    #   tree_str: dict mapping node_id -> Newick string
    
    # Use a larger distance matrix to avoid reindexing
    max_nodes = 2 * n - 1
    D_full = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D_full[:n, :n] = dist
    
    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    next_node_id = n
    
    while len(active) > 3:
        m = len(active)
        active_arr = np.array(active, dtype=np.int32)
        
        # Extract sub-matrix of distances for active nodes
        D_sub = D_full[np.ix_(active_arr, active_arr)]
        
        # Compute row sums
        r = D_sub.sum(axis=1)
        
        # Compute Q matrix: Q[a,b] = (m-2)*D[a,b] - r[a] - r[b]
        Q = (m - 2) * D_sub - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair with minimum Q
        min_val = Q.min()
        # Find indices of minimum
        min_flat = np.argmin(Q)
        min_i, min_j = np.unravel_index(min_flat, Q.shape)
        
        a_idx = min_i
        b_idx = min_j
        a = active[a_idx]
        b = active[b_idx]
        
        # Create a new internal node
        new_node = next_node_id
        next_node_id += 1
        
        # Compute branch lengths
        delta_a = 0.5 * (D_sub[a_idx, b_idx] + (r[a_idx] - r[b_idx]) / (m - 2))
        delta_b = D_sub[a_idx, b_idx] - delta_a
        
        # Clamp to non-negative
        delta_a = max(0.0, delta_a)
        delta_b = max(0.0, delta_b)
        
        # Build the subtree string
        tree_str[new_node] = f"({tree_str[a]}:{delta_a:.4f},{tree_str[b]}:{delta_b:.4f})"
        
        # Update distances from new_node to all remaining active nodes
        remaining = [x for k, x in enumerate(active) if k not in (a_idx, b_idx)]
        
        # For each remaining node x:
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        for x in remaining:
            new_d = 0.5 * (D_full[a, x] + D_full[b, x] - D_full[a, b])
            D_full[new_node, x] = new_d
            D_full[x, new_node] = new_d
        
        # Remove a and b from active, add new_node
        active = remaining + [new_node]
    
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
