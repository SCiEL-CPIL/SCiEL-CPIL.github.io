"""Reconstruct unrooted phylogenetic topology from simulated DNA sequences."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 260.0  # safety margin before 270s kill
_start_time = None


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    # Convert sequences to integer arrays for fast comparison
    seq_list = [sequences[lab] for lab in labels]
    L = len(seq_list[0])
    
    # Build integer representation: A=0, C=1, G=2, T=3
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    ints = np.array([[char_to_int[c] for c in s] for s in seq_list], dtype=np.int8)
    
    # Compute pairwise Hamming distances
    # For each pair, count positions where they differ
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Compare row i to all rows
        diff = (ints[i] != ints).sum(axis=1)
        dist[i] = diff / L
    
    # Apply JC correction
    # d = -3/4 * ln(1 - 4p/3), cap at 0.7499 to avoid log(0) or negative
    p = dist.copy()
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _neighbor_joining(dist):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a Newick string (unrooted, no branch lengths).
    """
    n = dist.shape[0]
    if n == 1:
        return f"t{n-1};"
    if n == 2:
        return f"(t{n-2},t{n-1});"
    if n == 3:
        return f"(t0,(t1,t2));"
    
    # Work with a mutable copy
    d = dist.copy()
    # Track which original taxon each current node represents (for Newick)
    # newick[i] holds the Newick subtree for current node i
    newick = [f"t{i}" for i in range(n)]
    
    # Active nodes: start with all n
    active = list(range(n))
    # We'll maintain a list of active node indices into a growing structure
    # Actually, NJ works by merging two nodes into one. Let's use a different approach:
    # Keep a list of "current" nodes. Each current node has an index in the d matrix.
    # When we merge a and b, we create a new node.
    
    # Simpler: use the standard NJ algorithm with a list of nodes
    # nodes: list of indices into the current distance matrix
    # But the matrix shrinks. Let's use a different strategy:
    # Maintain a list of "labels" (Newick strings) and a distance matrix between them.
    
    labels_list = [f"t{i}" for i in range(n)]
    dm = dist.copy()  # current distance matrix, shape (n, n)
    
    while len(labels_list) > 2:
        m = len(labels_list)
        
        # Compute row sums
        r = dm.sum(axis=1)  # r[i] = sum_j d(i,j)
        
        # Compute Q matrix: Q(i,j) = (m-2)*d(i,j) - r[i] - r[j]
        # Find the pair (i,j) with minimum Q
        best_q = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * dm[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        # Merge best_i and best_j
        a, b = best_i, best_j
        
        # Compute branch lengths (not strictly needed for topology, but good for correctness)
        # d_a = 0.5 * d(a,b) + (r[a] - r[b]) / (2*(m-2))
        # d_b = 0.5 * d(a,b) + (r[b] - r[a]) / (2*(m-2))
        # We don't need branch lengths for the topology-only Newick
        
        # Create new node
        new_label = f"({labels_list[a]},{labels_list[b]})"
        
        # Compute distances from new node to all other active nodes
        # d(new, k) = 0.5 * (d(a,k) + d(b,k) - d(a,b))
        other_indices = [i for i in range(m) if i not in (a, b)]
        new_dists = []
        for k in other_indices:
            new_d = 0.5 * (dm[a, k] + dm[b, k] - dm[a, b])
            new_dists.append(new_d)
        
        # Build new distance matrix
        new_m = m - 1
        new_dm = np.zeros((new_m, new_m), dtype=np.float64)
        new_labels = []
        
        # Copy distances among remaining nodes
        for i_idx, i_orig in enumerate(other_indices):
            new_labels.append(labels_list[i_orig])
            for j_idx, j_orig in enumerate(other_indices):
                if i_idx != j_idx:
                    new_dm[i_idx, j_idx] = dm[i_orig, j_orig]
        
        # Add new node distances
        new_labels.append(new_label)
        for i_idx, i_orig in enumerate(other_indices):
            new_dm[i_idx, new_m - 1] = new_dists[i_idx]
            new_dm[new_m - 1, i_idx] = new_dists[i_idx]
        
        labels_list = new_labels
        dm = new_dm
    
    # Final two nodes
    return f"({labels_list[0]},{labels_list[1]});"


def _fast_newick_from_splits(splits, n_taxa):
    """
    Given a set of splits (as frozensets of taxon indices), build a Newick string.
    This is a fallback; we primarily use NJ which gives us the tree directly.
    """
    # This is complex; we'll rely on NJ instead.
    raise NotImplementedError


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree from DNA sequences.
    Returns a Newick string with taxa t0..t(n-1).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a star tree as fallback
        n = len(sequences)
        labels = sorted(sequences, key=lambda x: int(x[1:]))
        return "(" + ",".join(labels) + ");"
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return f"{labels[0]};"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Compute JC-corrected distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    newick = _neighbor_joining(dist)
    
    return newick


# EVOLVE-BLOCK-END
