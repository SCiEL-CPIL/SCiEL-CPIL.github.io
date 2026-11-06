"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987) with NumPy acceleration.'''
import math
import time
import numpy as np


def _jc_distances(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy for speed."""
    n = len(labels)
    seqs = np.array([list(sequences[lab]) for lab in labels], dtype='U1')
    # Compute pairwise Hamming distances
    # For each pair, count mismatches
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(seqs[i] != seqs[j])
            p = mismatches / len(sequences[labels[0]])
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def _nj_newick(dist_matrix, labels, trees):
    """Run Neighbor-Joining and return Newick string."""
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(trees[lab] for lab in labels) + ");"
    
    # Work with indices
    nodes = list(range(n))
    # trees maps original label to its current Newick subtree
    # We'll track which original labels are in each node
    
    # dist_matrix is n x n
    # We'll maintain a dictionary mapping node_id -> (list of original labels it contains)
    # Actually, let's just track the current tree strings
    
    current_labels = list(labels)
    current_trees = {lab: lab for lab in labels}
    # dist_matrix is indexed by position in current_labels
    
    while len(current_labels) > 3:
        m = len(current_labels)
        # Compute row sums
        row_sums = np.sum(dist_matrix, axis=1)
        
        best_criterion = float('inf')
        best_i = 0
        best_j = 1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * dist_matrix[i, j] - row_sums[i] - row_sums[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j
        
        # Join best_i and best_j
        lab_i = current_labels[best_i]
        lab_j = current_labels[best_j]
        
        joined = f"({current_trees[lab_i]},{current_trees[lab_j]})"
        
        # Create new node
        new_label = f"n{m}_{lab_i}_{lab_j}"
        
        # Compute new distances
        new_dists = np.zeros(m - 1)
        rest_indices = [k for k in range(m) if k != best_i and k != best_j]
        for new_idx, old_idx in enumerate(rest_indices):
            new_dists[new_idx] = 0.5 * (dist_matrix[best_i, old_idx] + dist_matrix[best_j, old_idx] - dist_matrix[best_i, best_j])
        
        # Build new distance matrix
        new_matrix = np.zeros((m - 1, m - 1))
        for a, ia in enumerate(rest_indices):
            for b, ib in enumerate(rest_indices):
                new_matrix[a, b] = dist_matrix[ia, ib]
        for a, ia in enumerate(rest_indices):
            new_matrix[a, m - 2] = new_dists[a]
            new_matrix[m - 2, a] = new_dists[a]
        
        dist_matrix = new_matrix
        current_labels = [current_labels[ia] for ia in rest_indices] + [new_label]
        current_trees[new_label] = joined
    
    return "(" + ",".join(current_trees[lab] for lab in current_labels) + ");"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260  # 10s safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    dist_matrix = _jc_distances(sequences, labels)
    
    # Run NJ
    result = _nj_newick(dist_matrix, labels, {lab: lab for lab in labels})
    
    return result
# EVOLVE-BLOCK-END
