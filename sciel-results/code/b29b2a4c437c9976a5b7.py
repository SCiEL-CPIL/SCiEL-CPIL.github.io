"""FUAD-inspired Neighbor-Joining with vectorized distance computation and deadline guard."""
import time
import math
import numpy as np

_START_TIME = time.time()
# Leave a safety margin before the 270s wall-clock kill (contract says 180s CPU for all 20).
# We track elapsed wall-clock and stop if we are within ~10s of the limit.
_DEADLINE = _START_TIME + 260.0  # 10s safety margin


def _jc_distances(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using vectorized numpy ops."""
    n = len(labels)
    # Build a matrix of characters: each sequence is a string of A/C/G/T
    # Map to integers: A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    
    # Convert to numpy array of shape (n, L) with int8
    arr = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            arr[i, j] = char_map[c]
    
    # Compute pairwise Hamming distances
    # For each pair (i, j), count positions where arr[i] != arr[j]
    # Vectorized: for each position, compute all-pairs difference
    # This is O(n^2 * L) which for n=32, L=2000 is ~2M ops per position... 
    # Better: compute column-wise
    
    # Actually, let's do it more efficiently:
    # Create a one-hot encoding or use broadcasting
    # For n up to 32 and L up to 2000, we can do:
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    
    # For each position, compute which pairs differ
    # arr has shape (n, L). For position k, arr[:, k] has shape (n,)
    # We want, for each pair (i,j), the count of k where arr[i,k] != arr[j,k]
    
    # Efficient approach: for each position, compute the outer "not equal"
    # But that's L iterations of O(n^2) = 2000 * 1024 = 2M, fine.
    
    # Even better: use the fact that for 4 states, 
    # sum over positions of (arr[i,k] != arr[j,k]) 
    # = L - sum over positions of (arr[i,k] == arr[j,k])
    
    # Let's just do the direct loop over positions, vectorized over pairs
    # For each position k, create a (n, n) matrix of whether they differ
    # Sum over k.
    
    # Alternative: use numpy broadcasting
    # arr[:, np.newaxis, :] has shape (n, 1, L)
    # arr[np.newaxis, :, :] has shape (1, n, L)
    # difference = arr[:, np.newaxis, :] != arr[np.newaxis, :, :]  # shape (n, n, L)
    # dist_matrix = difference.sum(axis=2)
    # This creates an (n, n, L) array = 32*32*2000 = 2M ints, ~16MB, acceptable.
    
    diff = arr[:, np.newaxis, :] != arr[np.newaxis, :, :]
    hamming = diff.sum(axis=2).astype(np.float64)
    
    # Jukes-Cantor correction
    p = hamming / L
    # Handle p >= 0.75
    p_clipped = np.minimum(p, 0.7499)
    dist_matrix = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist_matrix, 0.0)
    
    return dist_matrix


def _neighbor_joining(dist_matrix, labels):
    """Run Neighbor-Joining on a distance matrix. Returns Newick string."""
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # dist_matrix is (n, n) numpy array
    # labels is list of strings
    # We'll maintain a list of "active" indices and a mapping from active index to Newick subtree
    active = list(range(n))
    subtree = {i: labels[i] for i in range(n)}
    D = dist_matrix.copy()
    
    while len(active) > 3:
        if time.time() > _DEADLINE:
            # Return a star tree as fallback
            return "(" + ",".join(labels) + ");"
        
        m = len(active)
        # Compute r_i = sum_j D[i,j] for j in active
        # D is full matrix, but we only care about active-active entries
        # For efficiency with small m, just compute directly
        r = np.zeros(m)
        for idx_i, i in enumerate(active):
            r[idx_i] = sum(D[i, j] for j in active if j != i)
        
        # Find pair (a, b) minimizing (m-2)*D[a,b] - r[a] - r[b]
        best_val = float('inf')
        best_a_idx = 0
        best_b_idx = 1
        for idx_a in range(m):
            for idx_b in range(idx_a + 1, m):
                a = active[idx_a]
                b = active[idx_b]
                val = (m - 2) * D[a, b] - r[idx_a] - r[idx_b]
                if val < best_val:
                    best_val = val
                    best_a_idx = idx_a
                    best_b_idx = idx_b
        
        a = active[best_a_idx]
        b = active[best_b_idx]
        
        # Compute branch lengths
        # delta = 0.5 * (D[a,b] + r[a] - r[b]) ... wait, standard NJ:
        # d(a, omega) = 0.5 * D[a,b] + (r[a] - r[b]) / (2*(m-2))
        # d(b, omega) = 0.5 * D[a,b] - (r[a] - r[b]) / (2*(m-2))
        # But for topology recovery (RF distance), branch lengths don't matter.
        # We just need the topology. So we can skip branch length computation.
        
        # New node
        new_node = len(subtree)  # next available index
        new_label = f"({subtree[a]},{subtree[b]})"
        
        # Compute distances from new node to all other active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        for x in active:
            if x != a and x != b:
                D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
                D[x, new_node] = D[new_node, x]
        
        # Remove a and b from active, add new_node
        active = [x for x in active if x != a and x != b]
        active.append(new_node)
        subtree[new_node] = new_label
        D[new_node, new_node] = 0.0
    
    # Final three nodes: create a star (unrooted)
    remaining = [subtree[x] for x in active]
    return "(" + ",".join(remaining) + ");"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences.
    
    Uses vectorized Jukes-Cantor distances and Neighbor-Joining.
    """
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 260.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    if time.time() > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    dist_matrix = _jc_distances(sequences, labels)
    
    # Run Neighbor-Joining
    result = _neighbor_joining(dist_matrix, labels)
    
    return result
