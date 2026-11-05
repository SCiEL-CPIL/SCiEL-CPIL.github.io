"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # Edge (u,v) crosses if labels[u] != labels[v]
    # Using matrix: cut = sum over u<v where labels differ of adjacency[u,v]
    # Efficient: for each vertex u, count neighbors with different label
    # cut = 0.5 * sum_u (sum_v adjacency[u,v] * (labels[u] != labels[v]))
    # But simpler: 
    # For vertex u with label l, crossing edges = sum of adjacency[u, v] where labels[v] != l
    n = len(labels)
    cross = np.zeros(n, dtype=np.int64)
    for u in range(n):
        if labels[u] == 0:
            cross[u] = adjacency[u, labels == 1].sum()
        else:
            cross[u] = adjacency[u, labels == 0].sum()
    return int(np.sum(cross) // 2)

def _local_search_flip(adjacency, labels, max_passes=50):
    """Single-flip local search. Returns improved labels and cut size."""
    n = len(labels)
    labels = labels.copy()
    # Precompute degree
    deg = adjacency.sum(axis=1).astype(np.int64)
    
    # For each vertex u, let s(u) = sum of signs of neighbors... 
    # Actually, for MaxCut: vertex u should be in set 0 if sum of neighbors in 0 > sum in 1
    # i.e., flip if same-side neighbors > other-side neighbors
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for u in range(n):
            same = adjacency[u, labels == labels[u]].sum()
            other = adjacency[u, labels != labels[u]].sum()
            if same > other:
                labels[u] = 1 - labels[u]
                improved = True
    return labels, _cut_size(adjacency, labels)

def _local_search_swap(adjacency, labels, max_passes=20):
    """Pairwise swap local search. Returns improved labels and cut size."""
    n = len(labels)
    labels = labels.copy()
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # For each pair (u,v) with u < v, check if swapping both improves cut
        # This is O(n^2) per pass, which for n=100 is 10000, acceptable
        for u in range(n):
            for v in range(u+1, n):
                if labels[u] == labels[v]:
                    continue  # Swapping two same-labeled vertices has no effect
                # Currently: u and v are in different sets
                # After swap: u goes to v's set, v goes to u's set
                # We need to compute the change in cut size
                # 
                # Let's compute the cut contribution change:
                # For vertex u: currently in set L_u, after swap in set 1-L_u
                # For vertex v: currently in set L_v, after swap in set 1-L_v
                # 
                # The change in cut for vertex u: 
                #   Current crossing from u: sum of adj[u,w] where labels[w] != labels[u]
                #   New crossing from u: sum of adj[u,w] where labels[w] != (1-labels[u])
                #   = sum of adj[u,w] where labels[w] == labels[u] (excluding v? no, v's label changes too)
                # 
                # This is complex because both u and v change. Let's compute directly.
                # 
                # Simpler: just try the swap and check if cut improves
                old_labels_u = labels[u]
                old_labels_v = labels[v]
                labels[u] = 1 - labels[u]
                labels[v] = 1 - labels[v]
                
                # Compute new cut size
                new_cut = _cut_size(adjacency, labels)
                # We need old cut... let's keep track
                # Actually, let's precompute old cut outside the loop
                # Let me restructure
                labels[u] = old_labels_u
                labels[v] = old_labels_v
                break  # This approach is too slow, let's use a different method
            break
        break
    # Fallback: just use flip-based search which is simpler
    return labels, _cut_size(adjacency, labels)

def _spectral_init(adjacency):
    """Use Fiedler vector of graph Laplacian for initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Compute eigenvalues and eigenvectors of L
    # We want the Fiedler vector (eigenvector of 2nd smallest eigenvalue)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # eigvecs are columns, sorted by eigenvalue ascending
        # Fiedler vector is eigvecs[:, 1]
        fiedler = eigvecs[:, 1]
        # Partition: vertices with positive Fiedler value -> 0, negative -> 1
        labels = (fiedler >= 0).astype(np.int8)
    except Exception:
        # Fallback to random
        labels = np.random.randint(0, 2, n).astype(np.int8)
    
    return labels

def _random_init(n, seed=None):
    """Random initial partition."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.randint(0, 2, n).astype(np.int8)

def partition(adjacency):
    """
    Maximum Cut via spectral initialization + multi-start local search.
    
    Parameters:
    - adjacency: bool matrix (n x n)
    
    Returns:
    - 0/1 label array of length n
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = len(adjacency)
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization
    spectral_labels = _spectral_init(adjacency)
    refined_labels, cut_val = _local_search_flip(adjacency, spectral_labels, max_passes=100)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined_labels.copy()
    
    # Check time
    if time.time() > deadline:
        return best_labels
    
    # Strategy 2: Multi-start random local search
    # Try several random starts
    max_starts = 20
    for i in range(max_starts):
        if time.time() > deadline - 5.0:
            break
        
        random_labels = _random_init(n, seed=42 + i)
        refined_labels, cut_val = _local_search_flip(adjacency, random_labels, max_passes=100)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
    
    # Final polish: more iterations on the best solution
    if best_labels is not None and time.time() < deadline - 2.0:
        polished_labels, cut_val = _local_search_flip(adjacency, best_labels, max_passes=200)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = polished_labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
