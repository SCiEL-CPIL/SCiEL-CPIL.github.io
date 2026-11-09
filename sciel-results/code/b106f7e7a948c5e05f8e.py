# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency, n):
    """Spectral partition using the Fiedler vector."""
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        A = adjacency.astype(np.float64)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        try:
            vals, vecs = eigsh(L, k=2, which='SA')
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        
        labels = (fiedler_vec > 0).astype(np.int8)
        
        if labels.sum() == 0 or labels.sum() == n:
            median = np.median(fiedler_vec)
            labels = (fiedler_vec > median).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
        
        return labels
    except Exception:
        return np.arange(n) % 2


def _local_search(adjacency, labels, deadline):
    """Iterative local search with running cut count and shuffled order."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        order = np.random.permutation(n)
        for v in order:
            if time.time() > deadline:
                break
            same_count = adjacency[v][labels == labels[v]].sum()
            other_count = adjacency[v][labels != labels[v]].sum()
            
            if same_count > other_count:
                labels[v] = 1 - labels[v]
                current_cut += (same_count - other_count)
                improved = True
    
    return labels, current_cut


def _walksat(adjacency, labels, deadline, max_iters=20000):
    """WalkSAT-style local search: greedy flips with random escape from local optima."""
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(adjacency, labels)
    
    total_degrees = adjacency.sum(axis=1)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate gains for all vertices
        diff = (labels[:, None] != labels[None, :])
        opp_counts = (adjacency & diff).sum(axis=1)
        gains = total_degrees - 2 * opp_counts
        
        best_gain = np.max(gains)
        
        if best_gain > 0:
            v = np.argmax(gains)
            labels[v] = 1 - labels[v]
            cut += best_gain
        else:
            # Local optimum. Pick a random uncut edge and flip one endpoint.
            uncut_mask = adjacency & (~diff)
            uncut_edges = np.argwhere(uncut_mask)
            
            if len(uncut_edges) == 0:
                break
            
            edge_idx = np.random.randint(0, len(uncut_edges))
            u, v = uncut_edges[edge_idx]
            flip_v = u if np.random.rand() < 0.5 else v
            
            cut += gains[flip_v]
            labels[flip_v] = 1 - labels[flip_v]
    
    return labels, cut


def _simulated_annealing(adjacency, labels, deadline, max_iters=30000):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = cut
    
    total_degrees = adjacency.sum(axis=1)
    current_temp = 5.0
    cooling_rate = 0.9997
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        diff = (labels[:, None] != labels[None, :])
        opp_counts = (adjacency & diff).sum(axis=1)
        gains = total_degrees - 2 * opp_counts
        
        v = np.random.randint(n)
        gain = gains[v]
        
        if gain > 0 or np.random.rand() < np.exp(gain / max(current_temp, 1e-8)):
            labels[v] = 1 - labels[v]
            cut += gain
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        current_temp *= cooling_rate
    
    return best_labels, best_cut


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n)


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    def _update_best(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Strategy 1: Spectral initialization + WalkSAT
    if n >= 2:
        spec_labels = _spectral_partition(adjacency, n)
        ws_labels, ws_cut = _walksat(adjacency, spec_labels, deadline)
        _update_best(ws_labels, ws_cut)
    
    # Strategy 2: Alternating + WalkSAT
    alt_labels = np.arange(n) % 2
    ws_labels, ws_cut = _walksat(adjacency, alt_labels, deadline)
    _update_best(ws_labels, ws_cut)
    
    # Strategy 3: Multiple random restarts with WalkSAT
    base_seed = 42
    for restart in range(20):
        if time.time() > deadline:
            break
        labels = _random_labels(n, base_seed + restart)
        ws_labels, ws_cut = _walksat(adjacency, labels, deadline)
        _update_best(ws_labels, ws_cut)
    
    # Strategy 4: Simulated annealing on the best result found so far
    if best_labels is not None and time.time() < deadline - 10:
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, deadline)
        _update_best(sa_labels, sa_cut)
    
    # Strategy 5: Final local search polish on the best result
    if best_labels is not None and time.time() < deadline - 3:
        ls_labels, ls_cut = _local_search(adjacency, best_labels, deadline)
        _update_best(ls_labels, ls_cut)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
