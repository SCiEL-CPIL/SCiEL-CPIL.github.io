# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _all_flip_gains(adj, labels):
    """Compute flip gains for all vertices vectorized.
    gain[v] = sum(adj[v, labels==labels[v]]) - sum(adj[v, labels!=labels[v]])
             = 2*sum(adj[v, labels==labels[v]]) - degree[v]
    """
    n = len(labels)
    degree = adj.sum(axis=1)
    # For each vertex v, count neighbors with same label
    # labels[:, None] == labels[None, :] gives same-label mask
    same_mask = (labels[:, np.newaxis] == labels[np.newaxis, :])
    same_counts = np.sum(adj * same_mask, axis=1)
    gains = 2.0 * same_counts - degree
    return gains


def _local_search_1flip(adj, labels, max_time=None):
    """Greedy 1-flip local search with vectorized gains."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        if max_time is not None and time.time() > max_time:
            return labels
        gains = _all_flip_gains(adj, labels)
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        improved = True
    return labels


def _two_swap_gain(adj, labels, i, j):
    """Compute the gain from simultaneously flipping vertices i and j.
    Only valid if i != j and labels[i] != labels[j] (otherwise no change).
    """
    if i == j or labels[i] == labels[j]:
        return 0
    gain_i = _flip_gain(adj, labels, i)
    gain_j = _flip_gain(adj, labels, j)
    # The edge (i,j) is counted in both gain_i and gain_j.
    # In combined flip, edge (i,j) stays crossing, so subtract 2*adj[i,j].
    return gain_i + gain_j - 2 * adj[i, j]


def _all_two_swap_gains(adj, labels):
    """Compute all 2-swap gains vectorized.
    Returns a matrix G where G[i,j] is the gain from flipping i and j.
    Only entries where labels[i] != labels[j] are meaningful.
    """
    n = len(labels)
    gains = _all_flip_gains(adj, labels)
    # G[i,j] = gains[i] + gains[j] - 2*adj[i,j] for i!=j, labels[i]!=labels[j]
    G = gains[:, np.newaxis] + gains[np.newaxis, :] - 2.0 * adj
    np.fill_diagonal(G, -np.inf)
    # Mask out pairs with same label
    same_mask = (labels[:, np.newaxis] == labels[np.newaxis, :])
    G[same_mask] = -np.inf
    return G


def _local_search_2swap(adj, labels, max_time=None):
    """2-swap local search: flip pairs of vertices with different labels."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        if max_time is not None and time.time() > max_time:
            return labels
        G = _all_two_swap_gains(adj, labels)
        # Find best positive gain
        max_gain = np.max(G)
        if max_gain <= 0:
            break
        i, j = np.unravel_index(np.argmax(G), G.shape)
        labels[i] = 1 - labels[i]
        labels[j] = 1 - labels[j]
        improved = True
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
        labels = _local_search_1flip(adj, labels, max_time)
        new_cut = _cut_size(adj, labels)
        if new_cut <= current_cut:
            labels = _local_search_2swap(adj, labels, max_time)
            new_cut = _cut_size(adj, labels)
            if new_cut <= current_cut:
                break
        current_cut = new_cut
    return labels


def _polish(adj, labels, max_time=None):
    """Final high-precision polish: aggressive 1-flip and 2-swap iterations."""
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 10
    n = len(labels)
    for _ in range(200):
        if time.time() > max_time:
            break
        # Multiple 1-flip passes
        for _ in range(10):
            if time.time() > max_time:
                break
            labels = _local_search_1flip(adj, labels, max_time)
        # 2-swap pass
        labels = _local_search_2swap(adj, labels, max_time)
        # Check convergence
        gains = _all_flip_gains(adj, labels)
        if np.max(gains) <= 0:
            G = _all_two_swap_gains(adj, labels)
            if np.max(G) <= 0:
                break
    return labels


def _tabu_search(adj, labels, tabu_len=7, max_iters=500, max_time=None):
    """Cerny tabu search for maximum cut with vectorized gains."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    current_cut = best_cut
    
    tabu_until = np.zeros(n, dtype=int)
    current_iter = 0
    
    for it in range(max_iters):
        if time.time() > max_time:
            break
        current_iter = it
        
        gains = _all_flip_gains(adj, labels)
        
        # Create mask for non-tabu vertices
        tabu_mask = (it < tabu_until)
        
        # Aspiration: allow tabu move if it improves best
        for v in range(n):
            if tabu_mask[v] and gains[v] > best_cut:
                tabu_mask[v] = False
        
        # Find best non-tabu gain
        masked_gains = np.where(tabu_mask, -np.inf, gains)
        best_gain = np.max(masked_gains)
        best_v = np.argmax(masked_gains)
        
        if best_gain == -np.inf:
            # All moves are tabu; pick best overall
            best_gain = np.max(gains)
            best_v = np.argmax(gains)
            if best_gain <= 0:
                break
        
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = current_iter + tabu_len
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _simulated_annealing(adj, labels, max_time=None, t_start=5.0, cooling=0.95, t_end=0.01):
    """Simulated annealing for maximum cut with vectorized gains."""
    n = len(labels)
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
    
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    current_cut = best_cut
    temperature = t_start
    
    np.random.seed(42)
    
    while temperature > t_end:
        if time.time() > max_time:
            break
        
        improved = False
        for v in range(n):
            if time.time() > max_time:
                break
            
            gain = _flip_gain(adj, labels, v)
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
            else:
                # Accept worse move with probability exp(gain / temperature)
                prob = np.exp(gain / temperature)
                if np.random.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    improved = True
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        temperature *= cooling
    
    return best_labels, best_cut


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (eigenvector of Laplacian)."""
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler > 0).astype(int)
    except:
        labels = np.arange(n) % 2
    return labels


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_labels = None
    best_cut = -1
    
    inits = []
    
    # Spectral
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Alternating
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # Random inits
    np.random.seed(42)
    for _ in range(8):
        rand_labels = np.random.randint(0, 2, n)
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
        
        ts_labels, ts_cut = _tabu_search(adj, labels.copy(), tabu_len=7, max_iters=200, max_time=deadline)
        
        # Do a final local search on the tabu result
        ts_labels = _full_local_search(adj, ts_labels, max_time=deadline)
        ts_cut = _cut_size(adj, ts_labels)
        
        if ts_cut > best_cut:
            best_cut = ts_cut
            best_labels = ts_labels.copy()
        
        # Apply simulated annealing from the best result so far
        if time.time() > deadline:
            break
        
        sa_labels, sa_cut = _simulated_annealing(adj, best_labels.copy(), max_time=deadline, t_start=5.0, cooling=0.95, t_end=0.01)
        
        # Do a final local search on the SA result
        sa_labels = _full_local_search(adj, sa_labels, max_time=deadline)
        sa_cut = _cut_size(adj, sa_labels)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Final polish pass on the best result
    if best_labels is not None and time.time() < deadline - 5:
        polished = _polish(adj, best_labels.copy(), max_time=deadline - 2)
        pol_cut = _cut_size(adj, polished)
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = polished.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
