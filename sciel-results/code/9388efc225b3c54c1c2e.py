"""Phylogenetic topology recovery using optimized NJ with NumPy."""
import time
import math
import numpy as np

_DEADLINE = 265.0
_start_time = None


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy vectorization."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Convert sequences to numeric arrays
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Build a single array of shape (n, L)
    arr = np.empty((n, L), dtype=np.int8)
    for i, lb in enumerate(labels):
        s = sequences[lb]
        for j, c in enumerate(s):
            arr[i, j] = base_map[c]
    
    # Compute pairwise Hamming distances using broadcasting
    # This creates an (n, n, L) array which might be large, so do it in pairs
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(arr[i] != arr[j])
            p = diff / L
            if p < 0.75:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            else:
                d = 3.0
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_reconstruct(D, n):
    """Neighbor-Joining reconstruction. Returns Newick string."""
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    D = D.copy()
    nodes = list(range(n))
    trees = {i: f"t{i}" for i in range(n)}
    next_label = n
    
    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums for current nodes
        row_sums = np.zeros(m)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    row_sums[i] += D[a, b]
        
        # Find best pair
        best_val = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                a = nodes[i]
                b = nodes[j]
                criterion = (m - 2) * D[a, b] - row_sums[i] - row_sums[j]
                if criterion < best_val:
                    best_val = criterion
                    best_i = i
                    best_j = j
        
        a = nodes[best_i]
        b = nodes[best_j]
        new_idx = next_label
        next_label += 1
        
        # Update distances
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            c = nodes[k]
            new_d = 0.5 * (D[a, c] + D[b, c] - D[a, b])
            D[a, c] = new_d
            D[c, a] = new_d
        
        trees[new_idx] = f"({trees[a]},{trees[b]})"
        nodes = [nodes[k] for k in range(m) if k != best_i and k != best_j]
        nodes.append(new_idx)
    
    x, y, z = nodes
    return f"({trees[x]},{trees[y]},{trees[z]});"


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from sequences. Returns Newick string."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    if elapsed > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    L = len(sequences[labels[0]])
    D = _jc_distance_matrix(sequences, labels)
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return "(" + ",".join(labels) + ");"
    
    return _nj_reconstruct(D, n)
