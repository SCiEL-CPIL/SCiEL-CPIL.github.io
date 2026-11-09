"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Cut size = sum over edges where labels differ
    # For a boolean adjacency matrix, this is 0.5 * sum(A * (labels[:,None] != labels[None,:]))
    # But more efficiently: for each vertex, count neighbors with different label
    n = len(labels)
    # Using matrix multiplication approach
    # cut = 0.5 * (sum of A * diff_matrix) where diff_matrix[i,j] = 1 if labels[i]!=labels[j]
    diff = (labels[:, None] != labels[None, :]).astype(np.float64)
    return 0.5 * np.sum(adj * diff)


def _local_search(adj, labels, max_iters=100, deadline=None):
    """Greedy local search: flip vertices if it improves the cut."""
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees for quick evaluation
    # For vertex v, if we flip it, the change in cut is:
    # new_cut - old_cut = (neighbors_in_opposite - neighbors_in_same)
    # where neighbors_in_opposite = number of neighbors of v with label != labels[v]
    # and neighbors_in_same = number of neighbors of v with label == labels[v]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for v in range(n):
            # Count neighbors with same and different labels
            same_count = np.sum(adj[v] & (labels == labels[v]))
            diff_count = np.sum(adj[v] & (labels != labels[v]))
            # If we flip v, same becomes diff and diff becomes same
            # Change in cut = diff_count - same_count
            if diff_count > same_count:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break
    
    return labels


def _random_labels(n, rng):
    """Generate random 0/1 labels."""
    return rng.integers(0, 2, size=n)


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline: 110 seconds to leave safety margin before 120s limit
    deadline = time.time() + 110
    
    # Use a fixed seed for reproducibility within the function call
    rng = np.random.default_rng(42)
    
    best_labels = None
    best_cut = -1
    
    # Multiple random restarts with local search
    # Each restart: random initialization + greedy local search
    while time.time() < deadline:
        # Check if we have enough time for at least one more restart
        # Estimate time per restart conservatively
        labels = _random_labels(n, rng)
        labels = _local_search(adj, labels, max_iters=50, deadline=deadline)
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Also try from a spectral-like initialization occasionally
        # Use random walk or power iteration for a better starting point
        # But for now, random restarts should suffice
    
    # Final pass: do one more local search on the best found
    if best_labels is not None:
        best_labels = _local_search(adj, best_labels, max_iters=50, deadline=deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
