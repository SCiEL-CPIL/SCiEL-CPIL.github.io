def _nj_step(dist, labels):
    """
    Perform one Neighbor-Joining step.
    Returns: (a_idx, b_idx, new_dist, new_labels)
    """
    n = len(labels)
    if n <= 3:
        return None, None, None, None
    
    # Compute Q matrix
    r = np.sum(dist, axis=1)  # row sums
    Q = (n - 2) * dist - np.outer(r, r)
    
    # Find minimum Q entry (off-diagonal)
    min_val = float('inf')
    min_i, min_j = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] < min_val:
                min_val = Q[i, j]
                min_i, min_j = i, j
    
    a, b = min_i, min_j
    
    # Create new label
    new_label = f"n{len(labels)}_{labels[a]}_{labels[b]}"
    
    rest = [k for k in range(n) if k not in (a, b)]
    new_labels = [labels[k] for k in rest] + [new_label]
    
    m = len(new_labels) # n-1
    new_dist = np.zeros((m, m), dtype=np.float64)
    
    # Fill distances between 'rest' nodes
    # Indices 0 to m-2 in new_dist correspond to 'rest'
    for idx_k, k in enumerate(rest):
        for idx_l, l in enumerate(rest):
            new_dist[idx_k, idx_l] = dist[k, l]
            
    # Fill distances from new node (index m-1) to 'rest' nodes
    for idx_k, k in enumerate(rest):
        d_uk = 0.5 * (dist[a, k] + dist[b, k] - dist[a, b])
        new_dist[idx_k, m-1] = d_uk
        new_dist[m-1, idx_k] = d_uk
        
    return a, b, new_dist, new_labels
