# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Vectorized: cut = sum of adj[i,j] where labels differ
    # = (1/2) * sum over all i,j of adj[i,j] * |labels[i] - labels[j]|
    # More efficient: 2*cut = sum_i sum_j adj[i,j] * (labels[i] != labels[j])
    # Use: cut = (n_pos * n_neg_total - sum of within-group edges) / 2
    # Actually simplest: 
    sign = 2.0 * labels - 1.0  # map 0->-1, 1->1
    # Edge (i,j) crosses iff sign[i]*sign[j] < 0
    # cut = (1/2) * sum_{i,j} adj[i,j] * (1 - sign[i]*sign[j]) / 2
    # = (1/4) * (sum_{i,j} adj[i,j] - sum_{i,j} adj[i,j]*sign[i]*sign[j])
    total = adj.sum()
    cross = np.dot(sign, np.dot(adj, sign))
    return (total - cross) / 4.0


def _all_flip_gains(adj, labels):
    """Compute flip gains for all vertices at once.
    gain[v] = (neighbors in same set) - (neighbors in opposite set)
    For vertex v with label l:
      same_count = sum_{u: labels[u]==l} adj[v,u]
      opp_count = sum_{u: labels[u]!=l} adj[v,u]
      gain[v] = same_count - opp_count
    
    Vectorized:
    Let s = 2*labels - 1 (so s in {-1, 1})
    For vertex v:
      same_count = sum_u adj[v,u] * (s[u] == s[v])
      opp_count = sum_u adj[v,u] * (s[u] != s[v])
    
    same_count[v] = sum_u adj[v,u] * (1 + s[v]*s[u]) / 2
    opp_count[v] = sum_u adj[v,u] * (1 - s[v]*s[u]) / 2
    gain[v] = same_count[v] - opp_count[v] = sum_u adj[v,u] * s[v]*s[u]
    """
    s = 2.0 * labels - 1.0
    # gain[v] = s[v] * sum_u adj[v,u] * s[u] = s[v] * (adj @ s)[v]
    adj_s = adj @ s
    gains = s * adj_s
    return gains


def _local_search_1flip(adj, labels, max_time=None):
    """Greedy 1-flip local search with vectorized gain computation."""
    n = len(labels)
    improved = True
    while improved:
        improved = False
        gains = _all_flip_gains(adj, labels)
        # Find all vertices with positive gain, flip the best one
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        # Flip all vertices with positive gain (or just the best one for safety)
        # Flip the best one to avoid issues
        best_v = int(np.argmax(gains))
        labels[best_v] = 1 - labels[best_v]
        improved = True
        if max_time is not None and time.time() > max_time:
            break
    return labels


def _local_search_2swap(adj, labels, max_time=None):
    """2-swap local search: flip pairs of vertices with different labels.
    Uses vectorized gain computation.
    """
    n = len(labels)
    improved = True
    while improved:
        improved = False
        if max_time is not None and time.time() > max_time:
            break
        
        gains = _all_flip_gains(adj, labels)
        # For a 2-swap of i and j (labels[i] != labels[j]):
        # combined_gain = gains[i] + gains[j] - 2*adj[i,j]
        # (because edge (i,j) is counted in both individual gains)
        
        # Find best pair: maximize gains[i] + gains[j] - 2*adj[i,j]
        # where labels[i] != labels[j]
        
        # Get indices by label
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        
        if len(idx0) == 0 or len(idx1) == 0:
            break
        
        # For each pair (i in idx0, j in idx1), compute gain
        # Vectorized approach:
        # gain_matrix[i,j] = gains[i] + gains[j] - 2*adj[i,j]
        # We want max over i in idx0, j in idx1
        
        g0 = gains[idx0]
        g1 = gains[idx1]
        
        # gain_matrix = g0[:,None] + g1[None,:] - 2*adj[np.ix_(idx0, idx1)]
        # Find max
        sub_adj = adj[np.ix_(idx0, idx1)]
        gain_matrix = g0[:, None] + g1[None, :] - 2.0 * sub_adj
        max_val = np.max(gain_matrix)
        
        if max_val <= 0:
            break
        
        pos = np.unravel_index(np.argmax(gain_matrix), gain_matrix.shape)
        i = idx0[pos[0]]
        j = idx1[pos[1]]
        
        labels[i] = 1 - labels[i]
        labels[j] = 1 - labels[j]
        improved = True
    
    return labels


def _full_local_search(adj, labels, max_time=None):
    """Alternate between 1-flip and 2-swap local search."""
    start_time = time.time()
    if max_time is None:
        max_time = start_time + 100
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
    
    for it in range(max_iters):
        if time.time() > max_time:
            break
        
        gains = _all_flip_gains(adj, labels)
        
        # Apply tabu: mask out tabu moves unless they beat best (aspiration)
        tabu_mask = (it < tabu_until)
        # Aspiration: allow tabu move if it improves on best
        aspiration = tabu_mask & (gains > best_cut)
        effective_tabu = tabu_mask & ~aspiration
        
        # Set tabu gains to -inf
        effective_gains = gains.copy()
        effective_gains[effective_tabu] = -np.inf
        
        if np.all(effective_gains == -np.inf):
            # All moves are tabu without aspiration; pick best non-tabu or break
            # Try without tabu
            best_v = int(np.argmax(gains))
        else:
            best_v = int(np.argmax(effective_gains))
        
        if gains[best_v] == -np.inf:
            break
        
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = it + tabu_len
        current_cut += gains[best_v]
        
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
    
    rng = np.random.RandomState(42)
    
    while temperature > t_end:
        if time.time() > max_time:
            break
        
        gains = _all_flip_gains(adj, labels)
        
        for v in range(n):
            if time.time() > max_time:
                break
            
            gain = gains[v]
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                # Recompute gains after flip (only affected vertices change,
                # but for simplicity recompute all)
                gains = _all_flip_gains(adj, labels)
            else:
                # Accept worse move with probability exp(gain / temperature)
                prob = np.exp(gain / temperature)
                if rng.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    gains = _all_flip_gains(adj, labels)
        
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
    rng = np.random.RandomState(42)
    for _ in range(8):
        rand_labels = rng.randint(0, 2, n)
        inits.append(rand_labels)
    
    # Phase 1: Quick local search on all inits to find good starting points
    local_results = []
    for init_idx, init_labels in enumerate(inits):
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        # Quick local search with time limit
        phase1_deadline = min(deadline, start_time + 30)
        labels = _full_local_search(adj, labels, max_time=phase1_deadline)
        cut = _cut_size(adj, labels)
        local_results.append((cut, labels.copy()))
    
    # Sort by cut size descending
    local_results.sort(key=lambda x: -x[0])
    
    # Phase 2: Apply tabu search on top candidates
    for cut_val, labels in local_results[:5]:
        if time.time() > deadline:
            break
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = labels.copy()
        
        # Tabu search from this local optimum
        ts_labels, ts_cut = _tabu_search(adj, labels.copy(), tabu_len=7, max_iters=300, max_time=deadline)
        
        # Do a final local search on the tabu result
        ts_labels = _full_local_search(adj, ts_labels, max_time=deadline)
        ts_cut = _cut_size(adj, ts_labels)
        
        if ts_cut > best_cut:
            best_cut = ts_cut
            best_labels = ts_labels.copy()
    
    # Phase 3: Simulated annealing from the best result
    if time.time() < deadline and best_labels is not None:
        sa_labels, sa_cut = _simulated_annealing(adj, best_labels.copy(), max_time=deadline, t_start=5.0, cooling=0.95, t_end=0.01)
        
        # Do a final local search on the SA result
        sa_labels = _full_local_search(adj, sa_labels, max_time=deadline)
        sa_cut = _cut_size(adj, sa_labels)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
