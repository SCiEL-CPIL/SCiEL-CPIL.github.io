"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Edge crosses if labels differ
    # For each edge (i,j) with adj[i,j]=1, count if labels[i] != labels[j]
    # Efficient: sum over i<j where adj[i,j]=1 and labels[i]!=labels[j]
    # Or: (adj @ (labels != labels) ) ... 
    # Simpler: for each vertex, count neighbors with different label, sum, divide by 2
    diff = labels != labels  # not useful directly
    # Use: for each i, count j where adj[i,j]=1 and labels[i]!=labels[j]
    # This counts each edge twice
    # cut = 0.5 * sum_i sum_j adj[i,j] * (labels[i] != labels[j])
    # In numpy: 
    label_diff = (labels[:, None] != labels[None, :]).astype(float)
    return 0.5 * np.sum(adj * label_diff)


def _local_search(adj, labels, deadline):
    """Greedy local search: flip vertex if it increases cut size."""
    n = len(labels)
    # Precompute degree contribution
    # For each vertex v, let s_v = sum of adj[v, u] for u with same label as v
    # and d_v = sum of adj[v, u] for u with different label
    # Flipping v changes cut by (d_v - s_v) = 2*d_v - degree_v
    # Actually: current contribution of v to cut is d_v (each edge to different label counted once in v's sum)
    # After flip, contribution becomes s_v
    # Change in total cut (which counts each edge once) = (s_v - d_v) / 1 ... 
    # Wait, let's be careful.
    # Total cut = 0.5 * sum_v sum_u adj[v,u] * [labels[v]!=labels[u]]
    # Contribution of vertex v to the sum (before /2) = sum_u adj[v,u] * [labels[v]!=labels[u]] = d_v
    # If we flip v, new contribution = sum_u adj[v,u] * [labels[v]==labels[u]] (old) = s_v
    # So change in the unhalved sum = s_v - d_v
    # Change in cut = (s_v - d_v) / 2
    # We want to flip if s_v > d_v
    
    # We'll iterate until no improvement or deadline
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            same = adj[v, labels == labels[v]].sum()
            other = adj[v, labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _random_restarts(adj, n, deadline, best_labels, best_cut):
    """Try multiple random initializations with local search."""
    rng = np.random.default_rng(42)
    count = 0
    while time.time() < deadline:
        # Random initialization
        labels = rng.integers(0, 2, size=n)
        # Local search
        labels = _local_search(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        count += 1
        if time.time() > deadline:
            break
    return best_labels, best_cut


def _spectral_init(adj, n):
    """Spectral initialization using Fiedler vector."""
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    # Find eigenvector for smallest non-zero eigenvalue
    # Use numpy's eigh for symmetric matrices
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is eigenvector corresponding to second smallest eigenvalue
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx[1]]
        # Threshold at median or zero
        labels = (fiedler > np.median(fiedler)).astype(int)
        return labels
    except:
        return np.arange(n) % 2


def partition(adjacency):
    """Give each vertex a 0/1 label to maximize cut size."""
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline: 110 seconds (10s safety margin before 120s limit)
    deadline = time.time() + 110
    
    # Start with spectral initialization
    best_labels = _spectral_init(adj, n)
    best_labels = _local_search(adj, best_labels, deadline)
    best_cut = _cut_size(adj, best_labels)
    
    # Also try alternating start
    alt_labels = np.arange(n) % 2
    alt_labels = _local_search(adj, alt_labels, deadline)
    alt_cut = _cut_size(adj, alt_labels)
    if alt_cut > best_cut:
        best_cut = alt_cut
        best_labels = alt_labels.copy()
    
    # Random restarts with remaining time
    if time.time() < deadline:
        best_labels, best_cut = _random_restarts(adj, n, deadline, best_labels, best_cut)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
