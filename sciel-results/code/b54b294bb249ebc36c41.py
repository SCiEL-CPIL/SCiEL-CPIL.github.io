"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Fast neighbour joining with vectorized distance computation and deadline guard.'''
import numpy as np
import time
import math


def _jc_distance_matrix(seq_array):
    """Compute pairwise JC-corrected distances from a 2D numpy array of character codes."""
    n = seq_array.shape[0]
    L = seq_array.shape[1]
    
    # Count mismatches using one-hot encoding for efficiency
    # seq_array is (n, L) with integer codes 0-3 (A=0, C=1, G=2, T=3)
    # Create one-hot: (n, L, 4)
    one_hot = np.zeros((n, L, 4), dtype=np.float32)
    one_hot[np.arange(n)[:, None], np.arange(L)[None, :], seq_array] = 1.0
    
    # For each position, compute pairwise matches
    # match_counts[i,j] = sum over positions of (seq[i,pos] == seq[j,pos])
    # Using one-hot: match at position p for pair (i,j) = sum_k one_hot[i,p,k] * one_hot[j,p,k]
    # This is: one_hot @ one_hot.T per position, summed
    # More efficiently: reshape and use matrix multiplication
    
    # Reshape to (n, L*4) and (n, L*4)
    oh_flat = one_hot.reshape(n, L * 4)
    # match_counts = oh_flat @ oh_flat.T / L  (since each position contributes 1 if same nucleotide)
    match_counts = (oh_flat @ oh_flat.T) / L
    
    # Mismatch proportion: p = 1 - match_counts
    p = 1.0 - match_counts
    
    # JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 by capping
    p_clipped = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    # Symmetrize
    d = (d + d.T) / 2.0
    
    return d


def _nj_tree(D, labels):
    """Run neighbour joining on distance matrix D with taxon labels. Returns Newick string."""
    n_taxa = len(labels)
    
    # Active nodes: indices into the original taxa
    active = list(range(n_taxa))
    trees = {i: labels[i] for i in range(n_taxa)}
    
    counter = 0
    while len(active) > 3:
        n = len(active)
        # Get current distance submatrix
        D_sub = D[np.ix_(active, active)]
        
        # Compute row sums (total distances)
        row_sums = D_sub.sum(axis=1)
        
        # Compute Q matrix: Q[i,j] = (n-2)*D[i,j] - row_sums[i] - row_sums[j]
        Q = (n - 2) * D_sub - row_sums[:, None] - row_sums[None, :]
        
        # Find the pair with minimum Q (excluding diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        
        # Ensure i < j for consistent naming
        if i > j:
            i, j = j, i
        
        a, b = active[i], active[j]
        
        # Create new node
        new_idx = -1 - counter
        counter += 1
        
        joined = f"({trees[a]},{trees[b]})"
        trees[new_idx] = joined
        
        # Compute distances from new node to all other active nodes
        rest_indices = [k for k in range(n) if k != i and k != j]
        for k in rest_indices:
            x = active[k]
            d_new_x = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[new_idx, x] = d_new_x
            D[x, new_idx] = d_new_x
        
        # Update active list
        rest = [active[k] for k in rest_indices]
        active = rest + [new_idx]
    
    # Build final Newick string
    return "(" + ",".join(trees[x] for x in active) + ");"


def reconstruct(sequences):
    # Deadline guard: 270s total budget, leave 30s safety margin
    deadline = time.time() + 240.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    # Convert sequences to numpy array
    seq_array = np.array([[ord(c) - ord('A') if c in 'ACGT' else 0 for c in sequences[label]] for label in labels])
    
    # Compute initial distance matrix
    D = _jc_distance_matrix(seq_array)
    
    # Run neighbour joining
    result = _nj_tree(D, labels)
    
    return result
# EVOLVE-BLOCK-END
