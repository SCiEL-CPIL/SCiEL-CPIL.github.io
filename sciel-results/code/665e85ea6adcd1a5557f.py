"""Neighbour Joining with Jukes-Cantor correction, vectorized for speed, plus deadline guard."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
# Global deadline guard: 270s wall-clock budget; leave a safety margin of ~15s.
_DEADLINE = time.time() + 265.0


def _time_left():
    return _DEADLINE - time.time()


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences p."""
    p = min(max(p, 0.0), 0.7499)
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix from DNA sequences using numpy."""
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    length = len(seqs[0])
    
    # Convert to integer arrays for fast comparison
    arrs = []
    for s in seqs:
        # Convert string to numpy array of integers
        # A=0, C=1, G=2, T=3
        lookup = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        arr_bytes = np.frombuffer(s.encode(), dtype=np.uint8)
        arr_int = lookup[arr_bytes]
        arrs.append(arr_int)
    
    # Build distance matrix
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(arrs[i] != arrs[j])
            p = diff / length
            d = _jc_distance(p)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _nj_tree(mat, labels):
    """Neighbour Joining on a distance matrix. Returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Active nodes
    active = list(range(n))
    D_act = mat.copy()
    # Subtrees as strings
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        if _time_left() < 5.0:
            # Build a star tree from remaining nodes as fallback
            return "(" + ",".join(trees[i] for i in active) + ");"
        
        m = len(active)
        # Row sums
        r = D_act.sum(axis=1)
        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D_act - r[:, None] - r[None, :]
        
        # Find the pair with minimum Q
        min_val = np.inf
        min_i = min_j = 0
        for i in range(m):
            for j in range(i + 1, m):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i = i
                    min_j = j
        
        i, j = min_i, min_j
        a = active[i]
        b = active[j]
        
        # Compute branch lengths (optional, but good for validity)
        d_ab = D_act[i, j]
        delta = (r[i] - r[j]) / (2 * (m - 2))
        la = max(0.0, 0.5 * d_ab + 0.5 * delta)
        lb = max(0.0, 0.5 * d_ab - 0.5 * delta)
        
        # Create new node
        new_id = next_id
        next_id += 1
        trees[new_id] = f"({trees[a]},{trees[b]})"
        
        # Update active list: remove i and j, add new node
        new_active = [x for k, x in enumerate(active) if k != i and k != j]
        new_active.append(new_id)
        
        # Update distance matrix
        if len(new_active) > 1:
            new_m = len(new_active)
            D_new = np.zeros((new_m, new_m), dtype=np.float64)
            
            # Build lookup: node_id -> index in old active list
            old_idx = {node: k for k, node in enumerate(active)}
            a_old = old_idx[a]
            b_old = old_idx[b]
            
            for k, x in enumerate(new_active):
                for l, y in enumerate(new_active):
                    if k == l:
                        continue
                    if x == new_id:
                        y_old = old_idx[y]
                        d_ay = D_act[a_old, y_old]
                        d_by = D_act[b_old, y_old]
                        D_new[k, l] = 0.5 * (d_ay + d_by - d_ab)
                    elif y == new_id:
                        x_old = old_idx[x]
                        d_ax = D_act[a_old, x_old]
                        d_bx = D_act[b_old, x_old]
                        D_new[k, l] = 0.5 * (d_ax + d_bx - d_ab)
                    else:
                        x_old = old_idx[x]
                        y_old = old_idx[y]
                        D_new[k, l] = D_act[x_old, y_old]
            
            D_act = D_new
        else:
            D_act = np.zeros((1, 1), dtype=np.float64)
        
        active = new_active
    
    # Build the final tree from the remaining 3 nodes
    if len(active) == 3:
        return "(" + ",".join(trees[i] for i in active) + ");"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    else:
        return trees[active[0]] + ";"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    mat = _build_distance_matrix(sequences, labels)
    
    # Run NJ
    return _nj_tree(mat, labels)
# EVOLVE-BLOCK-END
