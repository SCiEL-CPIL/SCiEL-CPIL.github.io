"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # For each vertex, count neighbors in the opposite partition
    # sum over v of sum over u where labels[u] != labels[v] of adj[v,u]
    # = sum over v of (degree[v] - sum over u where labels[u] == labels[v] of adj[v,u])
    # More efficiently: (labels @ adj @ (1 - labels)) / 2 ... but let's just do direct
    # Actually: cut = sum_{v < u} adj[v,u] * (labels[v] != labels[u])
    # Using matrix: cut = 0.5 * labels^T @ (D - A) @ (1 - 2*labels + 1) ... 
    # Simpler: for each v, count neighbors not in same set
    # cut = sum_v (adj[v] @ (1 - labels)) where labels are 0/1
    # But this double counts? No: adj[v] @ (1-labels) counts edges from v to opposite side.
    # Summing over all v gives 2 * cut_size (each edge counted twice).
    return int(np.sum(adj @ (1 - labels)) // 2)


def _spectral_init(adj, n, rng):
    """Generate initial partition using spectral method (Fiedler vector)."""
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    # Get eigenvalues and eigenvectors
    # For n=100, this is fast
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Split by median
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(np.int8)
        return labels
    except Exception:
        # Fallback to random
        return rng.randint(0, 2, size=n).astype(np.int8)


def _greedy_flip(adj, labels, n, max_iters=50):
    """Greedy 1-flip local search."""
    labels = labels.copy()
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for v in range(n):
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _two_swap(adj, labels, n, max_iters=20):
    """Local search with 2-swap moves."""
    labels = labels.copy()
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try all pairs of vertices in different sets
        set0 = np.where(labels == 0)[0]
        set1 = np.where(labels == 1)[0]
        for i in set0:
            for j in set1:
                # If we swap i and j, what's the gain?
                # Current contribution of i and j to cut:
                # i contributes: number of neighbors in set1
                # j contributes: number of neighbors in set0
                # After swap: i goes to set1, j goes to set0
                # New contribution of i: number of neighbors in set0 (excluding j)
                # New contribution of j: number of neighbors in set1 (excluding i)
                # This is complex; let's just compute cut before and after for small gains
                # Actually, let's use a simpler approach: just try the swap and check
                old_labels = labels.copy()
                labels[i] = 1
                labels[j] = 0
                new_cut = _cut_size(adj, labels)
                old_cut = _cut_size(adj, old_labels)
                if new_cut > old_cut:
                    improved = True
                    # Keep the swap
                else:
                    labels = old_labels
    return labels


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization and local search.
    adjacency: bool matrix (n x n)
    returns: 0/1 label per vertex (numpy array of int)
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for faster operations
    adj = np.array(adjacency, dtype=np.int8)
    
    # Deadline guard: return best result before 140s (safety margin from 150s)
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.RandomState(42)
    
    # Generate multiple candidate initial partitions
    best_labels = None
    best_cut = -1
    
    # 1. Spectral initialization
    labels_spec = _spectral_init(adj, n, rng)
    labels_spec = _greedy_flip(adj, labels_spec, n)
    cut_spec = _cut_size(adj, labels_spec)
    if cut_spec > best_cut:
        best_cut = cut_spec
        best_labels = labels_spec.copy()
    
    # 2. Random initializations with local search
    n_random_starts = 10
    for i in range(n_random_starts):
        if time.time() > deadline - 5:
            break
        labels_rand = rng.randint(0, 2, size=n).astype(np.int8)
        labels_rand = _greedy_flip(adj, labels_rand, n)
        cut_rand = _cut_size(adj, labels_rand)
        if cut_rand > best_cut:
            best_cut = cut_rand
            best_labels = labels_rand.copy()
    
    # 3. Apply 2-swap local search to the best solution
    if time.time() < deadline - 10:
        labels_2swap = _two_swap(adj, best_labels, n, max_iters=5)
        cut_2swap = _cut_size(adj, labels_2swap)
        if cut_2swap > best_cut:
            best_cut = cut_2swap
            best_labels = labels_2swap.copy()
    
    # 4. Additional greedy passes on the best solution
    if time.time() < deadline - 5:
        labels_final = _greedy_flip(adj, best_labels, n, max_iters=10)
        cut_final = _cut_size(adj, labels_final)
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final.copy()
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
