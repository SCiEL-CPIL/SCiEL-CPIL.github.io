"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Cut size = sum over edges (i,j) where labels[i] != labels[j]
    # Using matrix form: cut = 0.5 * sum_{i,j} adj[i,j] * (1 - labels[i]*labels[j]) if labels in {-1,1}
    # Or more directly:
    diff = labels[:, None] != labels[None, :]
    return int(np.sum(adj * diff))


def _flip_gain(adj, labels, v):
    """Compute the gain in cut size if we flip vertex v."""
    # Current cut contribution of v: number of neighbors with different label
    # After flip: number of neighbors with same label
    # Gain = (same_label_neighbors) - (diff_label_neighbors)
    #      = 2 * same_label_neighbors - degree(v)
    # Where same_label_neighbors = sum(adj[v, labels == labels[v]])
    same = np.sum(adj[v, labels == labels[v]])
    deg = np.sum(adj[v])
    return int(2 * same - deg)


def _local_search(adj, labels, max_passes=100, deadline=None):
    """Greedy local search: repeatedly flip vertices if it improves the cut."""
    n = len(labels)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if deadline is not None and time.time() > deadline:
            break
        for v in range(n):
            gain = _flip_gain(adj, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _pair_flip_search(adj, labels, max_passes=50, deadline=None):
    """Pair-flip local search: try flipping pairs of vertices to escape 1-flip local optima."""
    n = len(labels)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Compute gain of flipping both i and j
                # We need to compute the change in cut size when flipping both
                # Save original labels
                li = labels[i]
                lj = labels[j]
                
                # Temporarily flip both
                labels[i] = 1 - li
                labels[j] = 1 - lj
                
                new_cut = _cut_size(adj, labels)
                
                # Restore
                labels[i] = li
                labels[j] = lj
                
                old_cut = _cut_size(adj, labels)
                
                if new_cut > old_cut:
                    labels[i] = 1 - li
                    labels[j] = 1 - lj
                    improved = True
    return labels


def _spectral_init(adj, n):
    """Generate a partition using the Fiedler vector (second eigenvector of Laplacian)."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj.astype(float)
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # eigvals are sorted in ascending order by eigh
        fiedler = eigvecs[:, 1]
        # Partition: vertices with positive Fiedler component go to 0, negative to 1
        labels = (fiedler > 0).astype(np.int8)
        return labels
    except Exception:
        # Fallback to random
        return np.random.randint(0, 2, size=n).astype(np.int8)


def partition(adjacency):
    """Partition the graph to maximize the cut size."""
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline: 145 seconds to leave safety margin before 150s kill
    deadline = time.time() + 145.0
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # First restart: spectral initialization
            labels = _spectral_init(adj, n)
        elif restart == 1:
            # Second restart: alternating
            labels = np.arange(n) % 2
        else:
            # Subsequent restarts: random
            labels = np.random.randint(0, 2, size=n).astype(np.int8)
        
        # Local search with single flips
        labels = _local_search(adj, labels, max_passes=200, deadline=deadline)
        
        # Pair flip search to escape local optima
        labels = _pair_flip_search(adj, labels, max_passes=30, deadline=deadline)
        
        # Another round of single-flip local search after pair flips
        labels = _local_search(adj, labels, max_passes=100, deadline=deadline)
        
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we still have time, do additional random restarts
    while time.time() < deadline - 5.0:
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
        labels = _local_search(adj, labels, max_passes=100, deadline=deadline)
        labels = _pair_flip_search(adj, labels, max_passes=20, deadline=deadline)
        labels = _local_search(adj, labels, max_passes=50, deadline=deadline)
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
