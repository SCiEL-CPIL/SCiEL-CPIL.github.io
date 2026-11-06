import math
import time
import numpy as np

def _jc_distance(a, b):
    n = len(a)
    if n == 0: return 0.0
    a_arr = np.array(list(a), dtype='U1')
    b_arr = np.array(list(b), dtype='U1')
    diff = np.sum(a_arr != b_arr)
    p = diff / n
    if p >= 0.75: return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix(sequences, labels):
    n = len(labels)
    dist = np.zeros((n, n))
    seqs = [sequences[labels[i]] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(seqs[i], seqs[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _nj_tree(labels, dist):
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active_labels = list(labels)
    tree_parts = list(labels)
    
    while len(active_labels) > 3:
        m = len(active_labels)
        r = np.sum(dist, axis=1)
        Q = (m - 2) * dist - r[:, None] - r[None, :]
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        Q_masked = Q.copy()
        Q_masked[~mask] = np.inf
        min_val = np.min(Q_masked)
        min_idx = np.unravel_index(np.argmin(Q_masked), Q_masked.shape)
        i, j = min_idx
        d_ij = dist[i, j]
        new_label = f"({tree_parts[i]},{tree_parts[j]})"
        rest = [k for k in range(m) if k != i and k != j]
        new_dists = np.zeros(len(rest))
        for k, orig_idx in enumerate(rest):
            new_dists[k] = 0.5 * (dist[i, orig_idx] + dist[j, orig_idx] - d_ij)
        new_m = len(rest) + 1
        new_dist = np.zeros((new_m, new_m))
        if len(rest) > 0:
            rest_indices = np.array(rest)
            new_dist[:len(rest), :len(rest)] = dist[np.ix_(rest_indices, rest_indices)]
            new_dist[:len(rest), -1] = new_dists
            new_dist[-1, :len(rest)] = new_dists
        dist = new_dist
        tree_parts = [tree_parts[k] for k in rest] + [new_label]
        active_labels = rest + [m]
        
    return "(" + ",".join(tree_parts) + ");"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260.0
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 2: return f"({labels[0]},{labels[1]});"
        elif n == 1: return labels[0]
        else: return ""
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
        
    dist = _compute_distance_matrix(sequences, labels)
    newick = _nj_tree(labels, dist)
    
    # Time check
    if time.time() > deadline - 10:
        return newick
        
    # ML Refinement Placeholder
    # Given complexity, return NJ tree.
    return newick
