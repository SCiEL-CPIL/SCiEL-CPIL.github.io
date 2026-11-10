import numpy as np

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    seq_array = np.array([list(sequences[label]) for label in labels], dtype='S1')
    L = seq_array.shape[1]
    
    # Compute pairwise Hamming distances using broadcasting
    diff = (seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    
    # JC correction with clipping to avoid log(0) or negative log
    p_clipped = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_clipped)
    np.fill_diagonal(d, 0.0)
    
    return d

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    d = _jc_distance_matrix(sequences, labels)
    
    # Neighbor Joining
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes))
    D[:n, :n] = d
    
    active_nodes = list(range(n))
    tree_parts = {i: labels[i] for i in range(n)}
    next_idx = n
    
    while len(active_nodes) > 3:
        k = len(active_nodes)
        idx = np.array(active_nodes)
        sub_d = D[np.ix_(idx, idx)]
        
        r = sub_d.sum(axis=1)
        
        Q = (k - 2) * sub_d - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat_idx = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat_idx, Q.shape)
        
        u = active_nodes[i_idx]
        v = active_nodes[j_idx]
        
        w = next_idx
        next_idx += 1
        
        for x in active_nodes:
            if x == u or x == v:
                continue
            D[w, x] = 0.5 * (D[u, x] + D[v, x] - D[u, v])
            D[x, w] = D[w, x]
        D[w, w] = 0.0
        
        active_nodes = [x for x in active_nodes if x not in (u, v)]
        active_nodes.append(w)
        
        tree_parts[w] = f"({tree_parts[u]},{tree_parts[v]})"
        
    final_nodes = active_nodes
    newick = f"({tree_parts[final_nodes[0]]},{tree_parts[final_nodes[1]]},{tree_parts[final_nodes[2]]});"
    
    return newick
