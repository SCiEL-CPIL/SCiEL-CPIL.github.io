# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # For 0/1 labels, cut = 0.5 * sum_{v,u} A_{vu} * |labels[v] - labels[u]|
    # But more efficiently: cut = sum_v (adj[v] @ (labels != labels[v]))
    # Using the s-vector trick:
    # Let s = 2*labels - 1 (so s is in {-1, 1})
    # cut = 0.25 * (2*E - s^T A s) where E = total edges
    # Or simply: cut = 0.5 * sum(adj * |labels[:,None] - labels[None,:]|)
    # For n=100, direct computation is fine.
    n = len(labels)
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _all_flip_gains(adj, labels):
    """Compute the gain from flipping each vertex, vectorized."""
    n = len(labels)
    # gain[v] = same_count[v] - opp_count[v]
    # where same_count[v] = sum_u A_{vu} * (labels[u] == labels[v])
    # and opp_count[v] = sum_u A_{vu} * (labels[u] != labels[v])
    # same_count[v] = adj[v] @ (labels == labels[v])
    # opp_count[v] = adj[v] @ (labels != labels[v])
    # gain[v] = same_count[v] - opp_count[v]
    
    # Vectorized:
    # For each v, same_mask_v = (labels == labels[v])
    # same_counts = np.array([adj[v][labels == labels[v]].sum() for v in range(n)])
    # opp_counts = np.array([adj[v][labels != labels[v]].sum() for v in range(n)])
    # gains = same_counts - opp_counts
    
    # More efficient vectorized approach:
    # Let s = 2*labels - 1 (in {-1, 1})
    # gain[v] = sum_u A_{vu} * (1 if labels[u]==labels[v] else -1)
    # = sum_u A_{vu} * s_v * s_u
    # = (s * (A @ s))[v]
    s = 2 * labels - 1
    As = adj @ s
    gains = s * As
    return gains


def _local_search_1flip(adj, labels, deadline=None):
    """Improve partition by greedy 1-flip until no improvement."""
    n = len(labels)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        gains = _all_flip_gains(adj, labels)
        # Find the vertex with maximum positive gain
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        # Flip all vertices with positive gain (or just the best one?)
        # For safety, flip one at a time to avoid issues
        # Actually, flipping all with positive gain simultaneously can be problematic
        # because gains change after each flip. Let's flip the best one.
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        improved = True
    return labels


def _two_swap_gain(adj, labels, i, j):
    """Compute the gain from swapping labels of vertices i and j."""
    # Swapping labels of i and j means:
    # If labels[i] == labels[j], swapping has no effect (they're the same)
    # If labels[i] != labels[j], it's equivalent to flipping both i and j
    # Wait, "swap" here means exchanging their labels. If they have different labels,
    # swapping is the same as flipping both.
    # If they have the same label, swapping does nothing.
    
    li = labels[i]
    lj = labels[j]
    
    if li == lj:
        return 0.0
    
    # Flipping both i and j
    # The change in cut size from flipping vertex v is gain_v = same_count_v - opp_count_v
    # But when we flip two vertices simultaneously, the interaction matters.
    # Let's compute directly.
    
    # Before: cut contribution from edges involving i or j
    # After: cut contribution from edges involving i or j (with swapped labels)
    
    # Edges not involving i or j are unchanged.
    # Edges involving i or j may change.
    
    # Let's compute the change:
    # For each edge (i, u) where u != j:
    #   Before: contributes 1 if labels[i] != labels[u]
    #   After: contributes 1 if (1-labels[i]) != labels[u]
    #   Change: if labels[i] == labels[u]: 0 -> 1 (+1)
    #           if labels[i] != labels[u]: 1 -> 0 (-1)
    #   So change for edge (i,u) = 1 if labels[i]==labels[u] else -1
    #   = s_i * s_u where s = 2*labels-1
    #   Sum over u != j: s_i * sum_{u != j} A_{iu} s_u = s_i * (As_i - A_{ij} s_j)
    
    # Similarly for edges (j, u) where u != i:
    #   Change = s_j * (As_j - A_{ji} s_i)
    
    # Edge (i, j):
    #   Before: contributes 1 if labels[i] != labels[j] (which is true since li != lj)
    #   After: contributes 1 if (1-labels[i]) != (1-labels[j]) = labels[i] != labels[j] (still true)
    #   So no change for edge (i,j).
    
    s = 2 * labels - 1
    As = adj @ s
    
    # Sum of s_u * A_{iu} for all u
    sum_i = As[i]
    # Subtract the contribution from u=j
    sum_i_excl_j = sum_i - adj[i, j] * s[j]
    
    sum_j = As[j]
    sum_j_excl_i = sum_j - adj[j, i] * s[i]
    
    change_i = s[i] * sum_i_excl_j
    change_j = s[j] * sum_j_excl_i
    
    total_change = change_i + change_j
    return total_change


def _local_search_2swap(adj, labels, deadline=None):
    """Improve partition by greedy 2-swap until no improvement (vectorized)."""
    n = len(labels)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        s = 2 * labels - 1
        As = adj @ s
        
        # For pairs (i, j) with labels[i] != labels[j], the gain from swapping is:
        # change_i + change_j where:
        # change_i = s[i] * (As[i] - adj[i,j]*s[j])
        # change_j = s[j] * (As[j] - adj[j,i]*s[i])
        # = s[i]*As[i] + s[j]*As[j] - 2*adj[i,j]*s[i]*s[j]
        
        # Since labels[i] != labels[j], s[i]*s[j] = -1
        # So gain = s[i]*As[i] + s[j]*As[j] + 2*adj[i,j]
        
        # Precompute s[i]*As[i] for all i
        sAs = s * As
        
        # Only consider pairs with different labels
        # Build matrix of gains for all pairs
        # gain[i,j] = sAs[i] + sAs[j] + 2*adj[i,j] for i != j and labels[i] != labels[j]
        
        # Vectorized computation
        # sAs_sum[i,j] = sAs[i] + sAs[j]
        sAs_sum = sAs[:, np.newaxis] + sAs[np.newaxis, :]
        gain_matrix = sAs_sum + 2 * adj
        
        # Zero out diagonal and same-label pairs
        np.fill_diagonal(gain_matrix, -np.inf)
        
        # Zero out pairs where labels are the same
        same_label_mask = (labels[:, np.newaxis] == labels[np.newaxis, :])
        gain_matrix[same_label_mask] = -np.inf
        
        # Find the best pair
        max_gain = np.max(gain_matrix)
        if max_gain <= 0:
            break
        
        # Get the indices of the max gain
        idx = np.unravel_index(np.argmax(gain_matrix), gain_matrix.shape)
        i, j = idx
        if i > j:
            i, j = j, i
        
        labels[i], labels[j] = labels[j], labels[i]
        improved = True
    return labels


def _full_local_search(adj, labels, deadline=None):
    """Alternate 1-flip and 2-swap local search with more rounds."""
    if deadline is None:
        deadline = time.time() + 10
    labels = _local_search_1flip(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_2swap(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_1flip(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_2swap(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_1flip(adj, labels, deadline)
    return labels


def _tabu_search(adj, labels, tabu_len=10, max_iter=500, deadline=None):
    """Tabu search with 1-flip moves."""
    n = len(labels)
    if deadline is None:
        deadline = time.time() + 10
    
    current_labels = labels.copy()
    current_cut = _cut_size(adj, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    # Tabu list: store (vertex, iteration) for recently flipped vertices
    tabu = {}
    iteration = 0
    
    while iteration < max_iter:
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adj, current_labels)
        
        # Find best non-tabu move
        best_gain = -np.inf
        best_v = None
        for v in range(n):
            if v in tabu and tabu[v] > iteration - tabu_len:
                continue
            if gains[v] > best_gain:
                best_gain = gains[v]
                best_v = v
        
        if best_v is None:
            break
        
        # Aspiration: allow tabu move if it improves best
        if best_v in tabu and tabu[best_v] > iteration - tabu_len:
            if best_gain <= best_cut - current_cut:  # Not improving best
                # Try to find a non-tabu move
                found = False
                for v in range(n):
                    if v not in tabu or tabu[v] <= iteration - tabu_len:
                        if gains[v] > best_gain:
                            best_gain = gains[v]
                            best_v = v
                            found = True
                if not found:
                    break
        
        # Make the move
        current_labels[best_v] = 1 - current_labels[best_v]
        tabu[best_v] = iteration + 1
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
        
        iteration += 1
    
    return best_labels


def _simulated_annealing(adj, labels, deadline=None):
    """Simulated annealing with 1-flip and 2-swap moves."""
    n = len(labels)
    if deadline is None:
        deadline = time.time() + 10
    
    current_labels = labels.copy()
    current_cut = _cut_size(adj, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    temperature = 5.0
    cooling = 0.95
    
    while temperature > 0.01:
        if time.time() > deadline:
            break
        
        # Try a random 1-flip
        v = np.random.randint(n)
        gain = _all_flip_gains(adj, current_labels)[v]
        
        if gain > 0:
            current_labels[v] = 1 - current_labels[v]
            current_cut += gain
        else:
            if np.random.random() < np.exp(gain / temperature):
                current_labels[v] = 1 - current_labels[v]
                current_cut += gain
        
        # Occasionally try a 2-swap
        if np.random.random() < 0.3:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j and current_labels[i] != current_labels[j]:
                gain2 = _two_swap_gain(adj, current_labels, i, j)
                if gain2 > 0:
                    current_labels[i], current_labels[j] = current_labels[j], current_labels[i]
                    current_cut += gain2
                else:
                    if np.random.random() < np.exp(gain2 / temperature):
                        current_labels[i], current_labels[j] = current_labels[j], current_labels[i]
                        current_cut += gain2
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
        
        temperature *= cooling
    
    return best_labels


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
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Generate initial solutions
    inits = []
    
    # Spectral init
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Alternating init
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # Random inits
    np.random.seed(42)
    for _ in range(8):
        rand_labels = np.random.randint(0, 2, n)
        inits.append(rand_labels)
    
    for init_labels in inits:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # Full local search
        labels = _full_local_search(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Tabu search
        if time.time() > deadline:
            break
        tabu_labels = _tabu_search(adj, labels, tabu_len=10, max_iter=300, deadline=deadline)
        tabu_cut = _cut_size(adj, tabu_labels)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
        
        # Simulated annealing
        if time.time() > deadline:
            break
        sa_labels = _simulated_annealing(adj, labels, deadline=deadline)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Final polish: run more 1-flip and 2-swap iterations on the best so far
        if time.time() > deadline:
            break
        polished = _local_search_1flip(adj, best_labels.copy(), deadline)
        polished = _local_search_2swap(adj, polished, deadline)
        polished = _local_search_1flip(adj, polished, deadline)
        pol_cut = _cut_size(adj, polished)
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = polished.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
