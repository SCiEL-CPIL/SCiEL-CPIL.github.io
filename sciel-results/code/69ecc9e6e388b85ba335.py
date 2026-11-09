"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Use spectral method to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argmin(eigvals)
        v = eigvecs[:, idx]
        labels = (v > 0).astype(np.int32)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2


def _local_search(adjacency, labels, deadline, max_rounds=100):
    """Sequential local search: flip vertices one at a time if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Compute flip gains for all vertices efficiently
        # gain[v] = (num neighbors in opposite partition) - (num neighbors in same partition)
        # = 2 * (num neighbors in opposite partition) - degree[v]
        
        # For vertex v with label l, neighbors with label 1-l contribute to cut
        # gain = sum over neighbors u: (labels[u] != labels[v]) - (labels[u] == labels[v])
        #      = 2 * sum over neighbors u: (labels[u] != labels[v]) - degree[v]
        
        # Efficient computation:
        # Let A be adjacency matrix. For vertex v, 
        # num_opp = sum_u A[v,u] * (1 - (labels[v] == labels[u]))
        #          = sum_u A[v,u] - sum_u A[v,u] * (labels[v] == labels[u])
        #          = degree[v] - (A @ (labels == labels[v].repeat...))
        
        # Simpler: compute for each vertex the count of neighbors with same label
        same_counts = np.zeros(n, dtype=np.int64)
        for v in range(n):
            neighbors = np.where(adjacency[v])[0]
            if len(neighbors) > 0:
                same_counts[v] = np.sum(labels[neighbors] == labels[v])
        
        degrees = adjacency.sum(axis=1).astype(np.int64)
        gains = 2 * (degrees - same_counts) - degrees  # = degrees - 2*same_counts... wait
        
        # Let me recompute: 
        # If we flip v, edges that were crossing become non-crossing, and vice versa.
        # Current contribution to cut from v's edges: num_opp = degree[v] - same_counts[v]
        # After flip: num_opp_new = same_counts[v] (since same becomes opposite)
        # Delta = num_opp_new - num_opp = same_counts[v] - (degree[v] - same_counts[v]) = 2*same_counts[v] - degree[v]
        # We want to flip if delta > 0, i.e., 2*same_counts[v] > degree[v]
        
        gains = 2 * same_counts - degrees
        
        # Find vertices where flipping improves the cut
        flip_candidates = np.where(gains > 0)[0]
        
        if len(flip_candidates) == 0:
            break
        
        # Flip one vertex at a time (sequential) for stability
        # Pick the vertex with the largest gain
        best_v = flip_candidates[np.argmax(gains[flip_candidates])]
        labels[best_v] = 1 - labels[best_v]
        improved = True
    
    return labels


def _simulated_annealing(adjacency, labels, deadline, max_iters=2000):
    """Simulated annealing with vectorized gain computation."""
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees
    degrees = adjacency.sum(axis=1).astype(np.int64)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # Compute gain for flipping v
        neighbors = np.where(adjacency[v])[0]
        if len(neighbors) == 0:
            T *= cooling
            continue
        
        same_count = np.sum(labels[neighbors] == labels[v])
        delta = 2 * same_count - degrees[v]
        
        if delta > 0:
            # Accept improving move
            labels[v] = 1 - labels[v]
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept non-improving move with probability exp(delta/T)
            if T > 0.001 and np.random.random() < np.exp(delta / T):
                labels[v] = 1 - labels[v]
                current_cut += delta
        
        T *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 100  # 100s deadline with 20s safety margin
    
    # Get multiple diverse initializations
    initializations = []
    
    # Spectral initialization
    spec_labels = _spectral_partition(adjacency)
    initializations.append(spec_labels)
    
    # Random initializations
    for _ in range(2):
        rand_labels = np.random.randint(0, 2, size=n).astype(np.int32)
        if rand_labels.sum() > 0 and rand_labels.sum() < n:
            initializations.append(rand_labels)
    
    best_cut = -1
    best_labels = None
    
    # Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        
        ls_labels = _local_search(adjacency, init_labels, deadline)
        ls_cut = _cut_size(adjacency, ls_labels)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()
    
    # Simulated annealing with restarts to escape local optima
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            sa_labels = best_labels.copy()
        elif r == 1:
            # Perturb with small fraction
            sa_labels = best_labels.copy()
            num_perturb = max(1, n // 20)
            perturb_indices = np.random.choice(n, num_perturb, replace=False)
            sa_labels[perturb_indices] = 1 - sa_labels[perturb_indices]
        else:
            # Larger perturbation for later restarts
            sa_labels = best_labels.copy()
            num_perturb = max(1, n // (10 - r))
            perturb_indices = np.random.choice(n, num_perturb, replace=False)
            sa_labels[perturb_indices] = 1 - sa_labels[perturb_indices]
        
        sa_result, sa_cut = _simulated_annealing(adjacency, sa_labels, deadline)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_result.copy()
    
    # Final local search polish
    if time.time() < deadline:
        best_labels = _local_search(adjacency, best_labels, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
