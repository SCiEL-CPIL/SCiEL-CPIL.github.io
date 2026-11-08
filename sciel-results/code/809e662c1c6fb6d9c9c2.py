"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # For each vertex, count neighbors in the opposite partition.
    # Sum over all vertices and divide by 2 (each edge counted twice).
    n = len(labels)
    # labels is a 0/1 array
    # For vertex v with label L, its contribution is number of neighbors with label 1-L
    # We can compute this efficiently using matrix operations.
    # Let's create a matrix where entry (i,j) is 1 if labels differ
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    # Sum over all pairs where adjacency is 1 and labels differ
    # Since adjacency is symmetric, each edge is counted twice
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, max_time=None):
    """Iterated local search via single-flip improvement."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    # Precompute adjacency as int array for speed
    adj = adjacency.astype(np.int8)
    
    start_time = time.time()
    deadline = start_time + (max_time if max_time else 100)
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                return labels, current_cut
            
            # Count neighbors in same and opposite partition
            # Same partition: labels == labels[v]
            # Opposite partition: labels != labels[v]
            lv = labels[v]
            # Number of neighbors with label lv
            same_count = int((adj[v] & (labels == lv)).sum())
            # Number of neighbors with label 1-lv
            other_count = int((adj[v] & (labels == 1 - lv)).sum())
            
            # If flipping v increases the cut
            # Currently, edges from v to opposite partition contribute to cut
            # After flip, edges from v to same partition will contribute
            # So delta = same_count - other_count
            if same_count > other_count:
                labels[v] = 1 - lv
                current_cut += same_count - other_count
                improved = True
    
    return labels, current_cut


def partition(adjacency):
    """Find a good partition using multiple restarts and local search."""
    n = len(adjacency)
    start_time = time.time()
    total_deadline = start_time + 120  # 120 second limit
    
    best_labels = None
    best_cut = -1
    
    # Number of restarts to try
    # We'll do random restarts with local search
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > total_deadline - 5:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Run local search
        labels, cut = _local_search(adjacency, labels, max_time=total_deadline - time.time() - 2)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we've used up most of our time, stop
        if time.time() > total_deadline - 10:
            break
    
    # If no best found, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
