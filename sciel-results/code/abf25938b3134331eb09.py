"""Phylogenetic topology recovery using optimized Neighbor Joining with NumPy."""
import time
import math
import numpy as np

# Deadline guard: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0


def _time_exceeded():
    return time.time() > _DEADLINE


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)
    
    # Convert sequences to integer arrays: A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Use np.vectorize or manual loop for conversion, but then vectorize distance
    # Actually, converting to numpy arrays of bytes or ints is best.
    # Let's create a 2D array of shape (n, L) with values 0-3.
    seq_arrays = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        # This loop is O(n*L). For n=32, L=2000, this is 64k ops. Very fast.
        # We can optimize this further if needed, but it's likely not the bottleneck.
        for j, c in enumerate(s):
            seq_arrays[i, j] = char_map.get(c, 0)
            
    # Compute pairwise Hamming distances using broadcasting
    # seq_arrays is (n, L). We want (n, n) matrix of differences.
    # diff[i, j, k] = seq_arrays[i, k] != seq_arrays[j, k]
    # This creates a (n, n, L) boolean array. For n=32, L=2000, this is 32*32*2000 = 2M bools. ~2MB. Very manageable.
    diff = seq_arrays[:, np.newaxis, :] != seq_arrays[np.newaxis, :, :]
    # Sum over the last axis to get Hamming distances
    hamming = np.sum(diff, axis=2)
    
    # Convert to proportions
    p = hamming.astype(np.float64) / L
    
    # Apply JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (or close to it) to avoid log(0) or log(negative)
    # np.where handles this element-wise
    valid = p < 0.75
    D = np.zeros_like(p)
    D[valid] = -0.75 * np.log(1.0 - 4.0 * p[valid] / 3.0)
    D[~valid] = 3.0 # Cap at 3.0 (maximum JC distance)
    
    return D


def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string (topology only, no branch lengths needed for RF score).
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Work on a copy
    dist = D.copy()
    names = list(labels)
    
    # Track the tree structure: for each current node, store the Newick substring
    tree_parts = list(names)
    
    num_nodes = n
    while num_nodes > 3:
        if _time_exceeded():
            break
        
        m = num_nodes
        # Compute total distances (row sums)
        r = np.sum(dist, axis=1)
        
        # Compute NJ criterion matrix: Q[i, j] = (m - 2) * dist[i, j] - r[i] - r[j]
        # We only need to find the global minimum.
        # Create a matrix of the criterion
        # r_col = r[:, np.newaxis]
        # r_row = r[np.newaxis, :]
        # Q = (m - 2) * dist - r_col - r_row
        
        # To avoid creating the full Q matrix if not needed (though it's small),
        # we can just compute it. For m=32, it's 32x32.
        Q = (m - 2) * dist - np.outer(r, np.ones(m)) - np.outer(np.ones(m), r)
        
        # Set diagonal to infinity to ignore self-pairs
        np.fill_diagonal(Q, np.inf)
        
        # Find the index of the minimum value
        min_idx = np.argmin(Q)
        best_i, best_j = np.unravel_index(min_idx, (m, m))
        
        # Ensure i != j (should be guaranteed by inf diagonal, but safe)
        if best_i == best_j:
            continue # Should not happen
        
        i, j = best_i, best_j
        
        # Create new internal node
        new_part = "(" + tree_parts[i] + "," + tree_parts[j] + ")"
        
        # Compute distances from new node to all other nodes
        # d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
        # We need to update the distance matrix.
        # Remove rows/cols i and j.
        
        # Indices of nodes to keep
        keep = [k for k in range(m) if k != i and k != j]
        
        # Extract submatrix of kept nodes
        # new_dist will be (m-1) x (m-1)
        # The last row/col corresponds to the new node
        
        # 1. Get the distances between kept nodes
        # dist[np.ix_(keep, keep)]
        
        # 2. Compute new distances to the new node
        # new_d[k] = 0.5 * (dist[i, k] + dist[j, k] - dist[i, j])
        new_d = 0.5 * (dist[i, keep] + dist[j, keep] - dist[i, j])
        
        # Build new distance matrix
        new_dist = np.zeros((m - 1, m - 1), dtype=np.float64)
        # Fill the top-left (m-2)x(m-2) block
        new_dist[:m-2, :m-2] = dist[np.ix_(keep, keep)]
        # Fill the last column and row
        new_dist[:m-2, m-2] = new_d
        new_dist[m-2, :m-2] = new_d
        new_dist[m-2, m-2] = 0.0 # Self distance is 0
        
        # Update structures
        # The new node is at the end of the list
        tree_parts = [tree_parts[k] for k in keep] + [new_part]
        dist = new_dist
        num_nodes = m - 1
    
    # Combine remaining nodes
    if num_nodes == 3:
        result = "(" + ",".join(tree_parts) + ")"
    elif num_nodes == 2:
        result = "(" + ",".join(tree_parts) + ")"
    else:
        result = "(" + ",".join(tree_parts) + ")"
    
    return result + ";"


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string over labels t0..t(n-1).
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)
    
    return result
