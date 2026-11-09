# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _flip_gain(adj, labels, v):
    """Compute the gain from flipping vertex v.
    Gain = (neighbors in same set) - (neighbors in opposite set)
    """
    same_mask = (labels == labels[v])
    opp_mask = (labels != labels[v])
    same_count = adj[v][same_mask].sum()
    opp_count = adj[v][opp_mask].sum()
    return same_count - opp_count


def _two_swap_gain(adj, labels, i, j):
    """Compute the gain from simultaneously flipping vertices i and j.
    Only valid if i != j and labels[i] != labels[j] (otherwise no change).
    """
    if i == j or labels[i] == labels[j]:
        return 0
    gain_i = _flip_gain(adj, labels, i)
    gain_j = _flip_gain(adj, labels, j)
    return gain_i + gain_j - 2 * adj[i, j]


def _local_search_1flip(adj, labels, max_time=None):
    """Greedy 1-flip local search with vectorized gain computation."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        # Vectorized computation of flip gains
        # For each v: gain = sum_u adj[v,u] * (2*labels[v]*labels[u] - labels[v] - labels[u] + 1) ... 
        # Actually simpler: gain[v] = sum_u adj[v,u] * (1 - 2*|labels[v]-labels[u]|) * 2 - degree... no
        # Let's use: gain[v] = (neighbors in same) - (neighbors in opposite)
        # = sum_u adj[v,u] * (1 if labels[v]==labels[u] else -1)
        # = sum_u adj[v,u] * (2*labels[v]*labels[u] + 2*(1-labels[v])*(1-labels[u]) - 1)
        # = sum_u adj[v,u] * (2*labels[v]*labels[u] + 1 - 2*labels[v] - 2*labels[u] + 2*labels[v]*labels[u])
        # = sum_u adj[v,u] * (4*labels[v]*labels[u] - 2*labels[v] - 2*labels[u] + 1)
        # This is getting complex. Let's just use the matrix form:
        # gain[v] = sum_u adj[v,u] * (1 - 2*|labels[v]-labels[u]|)
        # But |labels[v]-labels[u]| is 1 if different, 0 if same.
        # So gain[v] = sum_u adj[v,u] * (1 - 2*(labels[v]!=labels[u]))
        # = sum_u adj[v,u] - 2*sum_u adj[v,u]*(labels[v]!=labels[u])
        
        # Let's compute the crossing count per vertex: cross[v] = sum_u adj[v,u] * (labels[v]!=labels[u])
        # Then gain[v] = degree[v] - 2*cross[v]
        
        diff = (labels[:, np.newaxis] != labels[np.newaxis, :]).astype(np.float64)
        cross = adj @ diff
        degree = adj.sum(axis=1)
        gains = degree - 2 * cross
        
        # Find the vertex with the largest positive gain
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        
        v = int(np.argmax(gains))
        labels[v] = 1 - labels[v]
        improved = True
        
        if max_time is not None and time.time() > max_time:
            break
    return labels


def _local_search_2swap(adj, labels, max_time=None):
    """2-swap local search: flip pairs of vertices with different labels."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        if max_time is not None and time.time() > max_time:
            break
        # Find all pairs with different labels
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                gain = _two_swap_gain(adj, labels, i, j)
                if gain > 0:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    improved = True
                    break
            if improved:
                break
    return labels


def _full_local_search(adj, labels, max_time=None):
    """Alternate between 1-flip and 2-swap local search."""
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    prev_cut = -1
    current_cut = _cut_size(adj, labels)
    for _ in range(50):
        if time.time() > max_time:
            break
        # 1-flip
        labels = _local_search_1flip(adj, labels, max_time)
        new_cut = _cut_size(adj, labels)
        if new_cut <= current_cut:
            # No improvement, try 2-swap
            labels = _local_search_2swap(adj, labels, max_time)
            new_cut = _cut_size(adj, labels)
            if new_cut <= current_cut:
                break
        current_cut = new_cut
    return labels


def _tabu_search(adj, labels, tabu_len=7, max_iters=500, max_time=None):
    """Cerny tabu search for maximum cut with vectorized gain computation."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    current_cut = best_cut
    
    # Tabu list: stores (vertex, iteration) of last flip
    tabu_until = np.zeros(n, dtype=int)
    current_iter = 0
    
    for it in range(max_iters):
        if time.time() > max_time:
            break
        current_iter = it
        
        # Vectorized computation of flip gains
        diff = (labels[:, np.newaxis] != labels[np.newaxis, :]).astype(np.float64)
        cross = adj @ diff
        degree = adj.sum(axis=1)
        gains = degree - 2 * cross
        
        # Mask tabu moves
        is_tabu = np.array([it < tabu_until[v] for v in range(n)])
        
        # Aspiration criterion: accept tabu move if it improves best
        # We'll handle this by temporarily allowing tabu moves that beat best_cut
        # For simplicity, just mask tabu moves and pick the best non-tabu
        # If all are tabu, pick the best one
        
        non_tabu_gains = np.where(is_tabu, -np.inf, gains)
        best_v = int(np.argmax(non_tabu_gains))
        best_gain = non_tabu_gains[best_v]
        
        if np.isinf(best_gain):
            # All moves are tabu, pick the best one anyway
            best_v = int(np.argmax(gains))
            best_gain = gains[best_v]
        
        # Perform the flip
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = current_iter + tabu_len
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (eigenvector of Laplacian)."""
    try:
        # Use sparse eigensolver for efficiency
        L = laplacian(csr_matrix(adj))
        # Find the second smallest eigenvalue and its eigenvector
        # For n=100, dense is fine, but let's try sparse
        eigvals, eigvecs = eigsh(L, k=2, which='SM')
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler > 0).astype(int)
    except:
        try:
            degree = adj.sum(axis=1)
            L = np.diag(degree) - adj
            eigvals, eigvecs = np.linalg.eigh(L)
            idx = np.argsort(eigvals)
            fiedler = eigvecs[:, idx[1]]
            labels = (fiedler > 0).astype(int)
        except:
            labels = np.arange(n) % 2
    return labels


def _goemans_maxwell_init(adj, n):
    """Goemans-Maxwell style initialization using random projections."""
    # Generate random vector
    rng = np.random.RandomState(42)
    v = rng.randn(n)
    # Project onto the null space of the degree matrix? No, just use as-is
    # The GM algorithm uses a random vector and then thresholding
    labels = (v > 0).astype(int)
    return labels


def _random_init(n, seed=None):
    """Random initialization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return rng.randint(0, 2, n)


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple restarts with different initializations
    # Use tabu search + local search hybrid
    
    inits = []
    
    # Spectral
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Alternating
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # Goemans-Maxwell style
    gm_labels = _goemans_maxwell_init(adj, n)
    inits.append(gm_labels)
    
    # Random inits with different seeds
    for seed in range(15):
        rand_labels = _random_init(n, seed=seed)
        inits.append(rand_labels)
    
    for init_idx, init_labels in enumerate(inits):
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # First, do local search to get to a local optimum
        labels = _full_local_search(adj, labels, max_time=deadline)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Then, apply tabu search starting from this local optimum
        if time.time() > deadline:
            break
        
        ts_labels, ts_cut = _tabu_search(adj, labels.copy(), tabu_len=7, max_iters=300, max_time=deadline)
        
        # Do a final local search on the tabu result
        ts_labels = _full_local_search(adj, ts_labels, max_time=deadline)
        ts_cut = _cut_size(adj, ts_labels)
        
        if ts_cut > best_cut:
            best_cut = ts_cut
            best_labels = ts_labels.copy()
    
    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
