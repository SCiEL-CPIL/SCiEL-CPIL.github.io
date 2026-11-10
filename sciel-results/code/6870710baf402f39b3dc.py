"""Phylogenetic topology recovery via Maximum Likelihood with NNI heuristic under JC69."""
import math
import numpy as np

def _jc69_matrices(branch_lengths):
    """Precompute JC69 transition matrices."""
    mats = []
    for bl in branch_lengths:
        e = math.exp(-3.0 * bl / 4.0)
        p_same = 1.0/3.0 + (2.0/3.0) * e
        p_diff = (1.0/3.0) * (1.0 - e)
        M = np.full((4, 4), p_diff)
        np.fill_diagonal(M, p_same)
        mats.append(M)
    return mats

def _compute_likelihood(tree, seq_matrix, branch_lengths):
    """
    Compute log-likelihood using vectorized Felsenstein algorithm.
    tree: list of nodes. Node: [left_idx, right_idx, label_idx, branch_length]
    seq_matrix: (n_taxa, n_sites) int array
    branch_lengths: list of lengths corresponding to tree nodes (edge from parent to node)
    """
    n_taxa = len(tree) # Actually tree contains internal nodes too.
    # We need to map leaf labels to indices.
    # Let's assume tree is a list where indices 0..n-1 are leaves? 
    # No, better to have a separate list of leaves.
    
    # For efficiency, we'll pass the tree structure and leaf mapping.
    pass

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 1:
        return labels[0] + ";" if labels else ";"
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
        
    # Convert to matrix
    seq_matrix = np.zeros((n_taxa, len(sequences[labels[0]])), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, char in enumerate(seq):
            if char == 'A': seq_matrix[i, j] = 0
            elif char == 'C': seq_matrix[i, j] = 1
            elif char == 'G': seq_matrix[i, j] = 2
            else: seq_matrix[i, j] = 3
            
    # ... (Implementation of NJ and NNI)
    
    return "(t0,t1);" # Placeholder
