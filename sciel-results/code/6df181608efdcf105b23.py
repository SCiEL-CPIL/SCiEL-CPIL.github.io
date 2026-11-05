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
    base_to_int = np.array([ord(c) for c in 'ACGT'], dtype=np.int8)
    # Map A->0, C->1, G->2, T->3
    # We'll encode each sequence as a matrix of 0/1 for each base
    # Then distance = sum of (s_i != s_j) across sites
    
    # Efficient approach: encode each base as one-hot, then compute Hamming distances
    # Or use a lookup table approach
    
    # Convert to integer arrays
    int_seqs = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            int_seqs[i, j] = ord(c)  # Use ASCII values for now
    
    # Compute pairwise Hamming distances efficiently
    # For each pair (i,j), count positions where int_seqs[i] != int_seqs[j]
    dist = np.zeros((n, n), dtype=np.float64)
    
    # Vectorized distance computation
    for i in range(n):
        # Compare sequence i with all sequences j > i
        diffs = (int_seqs[i] != int_seqs[i+1:]).sum(axis=1)
        p = diffs / L
        # Apply Jukes-Cantor correction
        # Handle p >= 0.75 case
        p_clipped = np.minimum(p, 0.7499)
        d_vals = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
        # For p >= 0.75, set to 3.0
        d_vals[p >= 0.75] = 3.0
        dist[i, i+1:] = d_vals
        dist[i+1:, i] = d_vals
    
    # Neighbor-Joining algorithm using arrays
    # active: list of current node indices
    # D: current distance matrix (reindexed to active nodes)
    # tree_str: dict mapping node_id -> Newick string
    
    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    next_node_id = n
    
    # Current distance matrix over active nodes
    # We'll maintain a 2D array indexed by position in active list
    D = dist.copy()
    
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums
        r = D.sum(axis=1)
        
        # Find the pair (i, j) that minimizes the NJ criterion:
        # Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        # Build Q matrix
        Q = (m - 2) * D - r[:, None] - r[None, :]
        
        # Find minimum Q value (off-diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx, j_idx = min_idx
        
        # Get the actual node ids
        a = active[i_idx]
        b = active[j_idx]
        
        # Create a new internal node joining a and b
        new_node = next_node_id
        next_node_id += 1
        
        # Compute branch lengths
        delta_a = 0.5 * (D[i_idx, j_idx] + (r[i_idx] - r[j_idx]) / (m - 2))
        delta_b = D[i_idx, j_idx] - delta_a
        
        # Ensure non-negative
        if delta_a < 0:
            delta_a = 0.0
        if delta_b < 0:
            delta_b = 0.0
        
        # Build the subtree string
        tree_str[new_node] = f"({tree_str[a]}:{delta_a:.4f},{tree_str[b]}:{delta_b:.4f})"
        
        # Update distances: D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        # Remove rows/cols i_idx and j_idx, add new row/col
        remaining_indices = [k for k in range(m) if k not in (i_idx, j_idx)]
        
        # New distance matrix
        new_D = np.zeros((m - 1, m - 1), dtype=np.float64)
        
        # Fill in the existing distances (excluding i_idx and j_idx)
        for new_i, old_i in enumerate(remaining_indices):
            for new_j, old_j in enumerate(remaining_indices):
                new_D[new_i, new_j] = D[old_i, old_j]
        
        # Add distances from new node to remaining nodes
        new_idx = m - 2  # Index of new node in new active list
        for new_i, old_i in enumerate(remaining_indices):
            new_d = 0.5 * (D[i_idx, old_i] + D[j_idx, old_i] - D[i_idx, j_idx])
            new_D[new_idx, new_i] = new_d
            new_D[new_i, new_idx] = new_d
        
        D = new_D
        
        # Update active list
        active = [active[k] for k in remaining_indices] + [new_node]
    
    # Final step: join the remaining 3 nodes into an unrooted tree
    a, b, c = active
    
    dab = D[0, 1]
    dac = D[0, 2]
    dbc = D[1, 2]
    
    # Branch lengths for 3-taxon tree
    len_a = max(0.0, (dab + dac - dbc) / 2.0)
    len_b = max(0.0, (dab + dbc - dac) / 2.0)
    len_c = max(0.0, (dac + dbc - dab) / 2.0)
    
    # Build final Newick
    result = f"({tree_str[a]}:{len_a:.4f},{tree_str[b]}:{len_b:.4f},{tree_str[c]}:{len_c:.4f})"
    
    return result
