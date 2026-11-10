import numpy as np
import time
import math

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_tree(d, labels):
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    dist = d.copy()
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    next_internal_id = n
    
    while len(active) > 3:
        n_act = len(active)
        r = np.zeros(n_act)
        for idx, i in enumerate(active):
            for j in active:
                if i != j:
                    r[idx] += dist[i, j]
        
        best_c = float('inf')
        best_i, best_j = -1, -1
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if i == j: continue
                c = (n_act - 2) * dist[i, j] - r[idx_i] - r[idx_j]
                if c < best_c:
                    best_c = c
                    best_i, best_j = i, j
                    
        i, j = best_i, best_j
        d_ij = dist[i, j]
        
        a = next_internal_id
        next_internal_id += 1
        
        for k in active:
            if k == i or k == j: continue
            d_ik = dist[i, k]
            d_jk = dist[j, k]
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            dist[a, k] = d_ak
            dist[k, a] = d_ak
            
        trees[a] = "(" + trees[i] + "," + trees[j] + ")"
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    return "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    
    # Use NJ for initial tree
    newick = _nj_tree(d, labels)
    
    return newick
