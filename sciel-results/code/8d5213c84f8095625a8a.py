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
    row_sums = A.sum(axis=1)
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
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
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler >= 0).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, n)
        return labels
    except Exception:
        return np.random.randint(0, 2, n)


def _adaptive_sa(adjacency, labels, t0, t_min, deadline, max_iter=10000):
    """
    Simulated annealing with:
      - vectorized gain computation
      - adaptive cooling based on acceptance ratio
      - 1-flip and 2-flip moves to escape deeper local minima
    Returns (best_labels, best_cut).
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    row_sums = A.sum(axis=1)
    
    current = labels.copy()
    current_cut = _cut_size(adjacency, current)
    best = current.copy()
    best_cut = current_cut
    
    T = t0
    # Track acceptance ratio for adaptive cooling
    accept_count = 0
    total_count = 0
    window_size = 200
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Adaptive cooling: adjust T based on recent acceptance rate
        if total_count >= window_size:
            ratio = accept_count / total_count
            # If too many rejections, cool faster; if too many acceptances, warm slightly
            if ratio < 0.1:
                T *= 0.9
            elif ratio > 0.5:
                T *= 1.05
            else:
                T *= 0.97
            T = max(T, t_min)
            accept_count = 0
            total_count = 0
        
        if T <= t_min:
            # At minimum temperature, only accept improvements
            pass
        
        # Choose move type: 1-flip (70%) or 2-flip (30%)
        use_two_flip = (np.random.random() < 0.3)
        
        if not use_two_flip:
            # 1-flip move
            v = np.random.randint(n)
            
            # Compute gain for flipping v
            # gain = 2 * (edges from v to same set) - row_sums[v]
            if current[v] == 0:
                same_mask = (current == 0)
            else:
                same_mask = (current == 1)
            same_mask[v] = False
            edges_to_same = A[v, same_mask].sum()
            delta = 2 * edges_to_same - row_sums[v]
            
            new_labels = current.copy()
            new_labels[v] = 1 - new_labels[v]
        else:
            # 2-flip move: pick two distinct vertices
            v1, v2 = np.random.choice(n, size=2, replace=False)
            
            # Compute combined gain for flipping both v1 and v2
            # Current cut contribution from v1: edges to opposite set
            # Current cut contribution from v2: edges to opposite set
            # After flipping both, need to recompute
            
            # Simpler: just compute cut before and after
            new_labels = current.copy()
            new_labels[v1] = 1 - new_labels[v1]
            new_labels[v2] = 1 - new_labels[v2]
            new_cut = _cut_size(adjacency, new_labels)
            delta = new_cut - current_cut
        
        total_count += 1
        
        # Accept/reject
        if delta > 0:
            current = new_labels
            current_cut = current_cut + delta
            if current_cut > best_cut:
                best_cut = current_cut
                best = current.copy()
            accept_count += 1
        else:
            if T > t_min:
                # Boltzmann acceptance
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current = new_labels
                    current_cut = current_cut + delta
                    accept_count += 1
    
    return best, best_cut


def partition(adjacency):
    """Find maximum cut using spectral init + adaptive SA + local search."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = len(adjacency)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + local search
    if time.time() < deadline - 20:
        labels = _spectral_partition(adjacency)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 2: Random initialization + local search
    if time.time() < deadline - 20:
        labels = np.random.randint(0, 2, n)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 3: Adaptive SA from best current state
    if best_labels is not None and time.time() < deadline - 15:
        sa_labels, sa_cut = _adaptive_sa(
            adjacency, best_labels, t0=5.0, t_min=0.01,
            deadline=deadline - 10, max_iter=20000
        )
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Polish with local search
        if time.time() < deadline - 5:
            best_labels = _local_search(adjacency, best_labels)
            best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 4: Multiple random restarts with adaptive SA
    restarts = 0
    while time.time() < deadline - 15 and restarts < 8:
        labels = np.random.randint(0, 2, n)
        labels = _local_search(adjacency, labels)
        cut = _cut_size(adjacency, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Run adaptive SA from this local optimum
        if time.time() < deadline - 10:
            sa_labels, sa_cut = _adaptive_sa(
                adjacency, labels, t0=3.0, t_min=0.01,
                deadline=deadline - 10, max_iter=15000
            )
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
            
            # Final polish
            if time.time() < deadline - 5:
                best_labels = _local_search(adjacency, best_labels)
                best_cut = _cut_size(adjacency, best_labels)
        
        restarts += 1
    
    # Final polish pass
    if best_labels is not None and time.time() < deadline - 2:
        best_labels = _local_search(adjacency, best_labels)
        best_cut = _cut_size(adjacency, best_labels)
    
    return best_labels
# EVOLVE-BLOCK-END
