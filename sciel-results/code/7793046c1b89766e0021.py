<<<<<<< SEARCH
def _vectorized_local_search(adjacency, labels, deadline, max_rounds=100):
    """
    Vectorized local search that computes all gains at once and flips all positive gains.
    This is faster per round than sequential but may oscillate; use fewer rounds.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    degree = A.sum(axis=1).astype(np.int32)
    
    current_cut = _cut_size_fast(adjacency, labels)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Compute gain for each vertex in one vectorized pass
        # gain[v] = degree[v] - 2 * opp_count[v]
        # opp_count[v] = number of neighbors with label != labels[v]
        
        # For each v, opp_count[v] = sum_j A[v,j] * (labels[j] != labels[v])
        # We can compute this as: for label 0 vertices, opp_count = sum of A[v, label1]
        # For label 1 vertices, opp_count = sum of A[v, label0]
        
        label0_mask = (labels == 0)
        label1_mask = (labels == 1)
        
        # For vertices with label 0: opp_count = A @ label1_mask
        # For vertices with label 1: opp_count = A @ label0_mask
        
        As_label1 = A @ label1_mask.astype(np.int32)
        As_label0 = A @ label0_mask.astype(np.int32)
        
        opp_counts = np.where(label0_mask, As_label1, As_label0)
        gains = degree - 2 * opp_counts
        
        # Flip all vertices with positive gain
        flip_mask = gains > 0
        if not flip_mask.any():
            break
        
        # Update cut: sum of gains for flipped vertices
        current_cut += int(gains[flip_mask].sum())
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels, current_cut
=======
def _vectorized_local_search(adjacency, labels, deadline, max_rounds=100):
    """
    Vectorized local search that computes all gains at once and flips all positive gains.
    Uses a more stable approach: only flip if gain > 0, and use a threshold to avoid oscillation.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    degree = A.sum(axis=1).astype(np.int32)
    
    current_cut = _cut_size_fast(adjacency, labels)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        label0_mask = (labels == 0)
        label1_mask = (labels == 1)
        
        As_label1 = A @ label1_mask.astype(np.int32)
        As_label0 = A @ label0_mask.astype(np.int32)
        
        opp_counts = np.where(label0_mask, As_label1, As_label0)
        gains = degree - 2 * opp_counts
        
        # Only flip vertices with strictly positive gain
        flip_mask = gains > 0
        if not flip_mask.any():
            break
        
        current_cut += int(gains[flip_mask].sum())
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels, current_cut
>>>>>>> REPLACE

