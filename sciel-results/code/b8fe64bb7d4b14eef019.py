"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # An edge (i,j) crosses if labels[i] != labels[j]
    # Using: cut = sum over all i<j where labels differ of adj[i,j]
    # Vectorized: (labels[:, None] != labels[None, :]) & adj, then sum and divide by 2 (since each edge counted twice)
    mask = (labels[:, None] != labels[None, :]) & adj
    return mask.sum() // 2


def _greedy_improve(adj, labels, deadline):
    """Greedy local improvement: flip vertices if it increases cut size."""
    n = len(labels)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            # Count neighbors in same group vs other group
            same_count = adj[v][labels == labels[v]].sum()
            other_count = adj[v][labels != labels[v]].sum()
            if other_count > same_count:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Greedy local improvement
        labels = _greedy_improve(adj, labels, deadline)
        
        if time.time() > deadline:
            break
        
        # Evaluate this solution
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we never got a chance to run, use alternating start
    if best_labels is None:
        labels = np.arange(n) % 2
        best_labels = labels
    
    return best_labels
# EVOLVE-BLOCK-END
