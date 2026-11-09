"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    # Precompute row sums for efficiency
    row_sums = A.sum(axis=1)
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            # Compute gain if we flip vertex v
            # Current contribution: edges from v to nodes in opposite set
            # After flip: edges from v to nodes in same set become crossing
            # Gain = (edges to same set) - (edges to opposite set)
            #      = 2*(edges to same set) - row_sums[v]
            
            if labels[v] == 0:
                same_set = (labels == 0)
                same_set[v] = False
            else:
                same_set = (labels == 1)
                same_set[v] = False
            
            edges_to_same = A[v, same_set].sum()
            gain = 2 * edges_to_same - row_sums[v]
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels


def _spectral_partition(adjacency):
    """Use spectral method to get initial partition via Fiedler vector."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A  # Laplacian
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is eigenvector corresponding to second smallest eigenvalue
        # Sort eigenvalues and get the second one
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        # Threshold at 0
        labels = (fiedler >= 0).astype(int)
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, n)
        return labels
    except Exception:
        return np.random.randint(0, 2, n)


def _simulated_annealing(adjacency, labels, t0, tf, alpha, max_iter, deadline):
    """Simulated annealing to escape local optima."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    row_sums = A.sum(axis=1)
    
    current_labels = labels.copy()
    current_cut = _cut_size(adjacency, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        new_labels = current_labels.copy()
        new_labels[v] = 1 - new_labels[v]
        
        # Compute new cut size
        new_cut = _cut_size(adjacency, new_labels)
        delta = new_cut - current_cut
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta > 0:
            current_labels = new_labels
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = new_labels.copy()
        else:
            T = t0 * (alpha ** it)
            if T > 1e-10:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_labels = new_labels
                    current_cut = new_cut
    
    return best_labels, best_cut


def partition(adjacency):
    """Find maximum cut using spectral initialization + simulated annealing + local search."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin for 150s limit
    
    n = len(adjacency)
    
    # Try multiple strategies and keep the best
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + local search
    if time.time() < deadline - 30:
        labels = _spectral_partition(adjacency)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Strategy 2: Random initialization + local search
    if time.time() < deadline - 30:
        labels = np.random.randint(0, 2, n)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Strategy 3: Simulated annealing from best current
    if best_labels is not None and time.time() < deadline - 15:
        sa_labels, sa_cut = _simulated_annealing(
            adjacency, best_labels, t0=10.0, tf=0.1, alpha=0.99, 
            max_iter=5000, deadline=deadline - 10
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        # Final local search on SA result
        if time.time() < deadline - 5:
            best_labels = _local_search(adjacency, best_labels)
            best_cut = _cut_size(adjacency, best_labels)
    
    # Strategy 4: More SA restarts if time permits
    restarts = 0
    while time.time() < deadline - 10 and restarts < 5:
        labels = np.random.randint(0, 2, n)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        sa_labels, sa_cut = _simulated_annealing(
            adjacency, labels, t0=5.0, tf=0.05, alpha=0.995,
            max_iter=3000, deadline=deadline - 10
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        if time.time() < deadline - 5:
            best_labels = _local_search(adjacency, best_labels)
            best_cut = _cut_size(adjacency, best_labels)
        
        restarts += 1
    
    return best_labels
# EVOLVE-BLOCK-END
