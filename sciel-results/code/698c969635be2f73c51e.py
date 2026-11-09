"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[None, :] != labels[:, None])
    return int((adj * diff).sum() / 2)


def _fast_local_search(adj, labels, deadline):
    """
    Optimized local search for Max-Cut using gain array updates.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degree
    deg = adj.sum(axis=1)
    
    # Initialize gain array
    # gain[v] = (# neighbors in opposite set) - (# neighbors in same set)
    # If gain[v] > 0, flipping v increases cut.
    diff_mask = (labels[None, :] != labels[:, None])
    diff_counts = (adj * diff_mask).sum(axis=1)
    gains = 2 * diff_counts - deg
    
    improved = True
    while improved:
        improved = False
        # Find the vertex with the maximum positive gain
        max_gain_idx = np.argmax(gains)
        max_gain_val = gains[max_gain_idx]
        
        if max_gain_val <= 0:
            break
        
        v = max_gain_idx
        # Flip v
        labels[v] = 1 - labels[v]
        improved = True
        
        # Update gains for neighbors of v
        neighbors = np.where(adj[v] == 1)[0]
        
        for u in neighbors:
            # Update diff_counts[u]
            # If labels[u] != labels[v] (current), they are different.
            # Before flip, v had opposite label, so they were same.
            # So diff_count[u] increases by 1.
            # If labels[u] == labels[v] (current), they are same.
            # Before flip, they were different.
            # So diff_count[u] decreases by 1.
            
            if labels[u] != labels[v]:
                diff_counts[u] += 1
            else:
                diff_counts[u] -= 1
            
            gains[u] = 2 * diff_counts[u] - deg[u]
        
        # Update diff_counts[v] and gains[v]
        # Since v flipped, its relationship with all neighbors flipped.
        # new_diff_count[v] = deg[v] - old_diff_count[v]
        # We don't have old_diff_count[v] directly, but we can recompute.
        # Recomputing is O(deg(v)) which is small.
        diff_counts[v] = (adj[v] * (labels != labels[v])).sum()
        gains[v] = 2 * diff_counts[v] - deg[v]
        
        if time.time() > deadline:
            break
            
    return labels


def _random_labels(n, seed=None):
    """Generate random partition labels."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.randint(0, 2, size=n).astype(np.int8)


def partition(adjacency):
    """Main partition function using multi-restart fast local search."""
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_labels = None
    best_cut = -1
    
    max_restarts = 500
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        labels = _random_labels(n)
        improved_labels = _fast_local_search(adj, labels, deadline)
        cut = _cut_size(adj, improved_labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = improved_labels.copy()
            
    if best_labels is None:
        return _random_labels(n)
        
    return best_labels
# EVOLVE-BLOCK-END
