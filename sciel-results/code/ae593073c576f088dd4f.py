"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)


def _flip_gain(adj, labels, v):
    """Compute the gain in cut size if we flip vertex v."""
    same_side = (labels == labels[v]).astype(np.int8)
    other_side = (labels != labels[v]).astype(np.int8)
    count_same = int(adj[v] @ same_side)
    count_other = int(adj[v] @ other_side)
    return count_same - count_other


def _two_flip_gain(adj, labels, v, w):
    """Compute the gain in cut size if we flip both vertices v and w."""
    # We need to compute the change in cut size when flipping both v and w.
    # Let's compute it by checking all neighbors of v and w.
    # For each neighbor u of v (u != w):
    #   if labels[u] == labels[v]: edge (v,u) was NOT in cut, after flip v it IS in cut, after flip w (if u==w, special case)
    # This is complex. Let's just compute the cut before and after.
    old_cut = _cut_size(adj, labels)
    new_labels = labels.copy()
    new_labels[v] = 1 - new_labels[v]
    new_labels[w] = 1 - new_labels[w]
    new_cut = _cut_size(adj, new_labels)
    return new_cut - old_cut


def _local_search(adj, labels, deadline, use_two_flips=False):
    """Greedy local search: repeatedly flip vertices that improve the cut until no improvement."""
    n = len(labels)
    labels = labels.copy()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            gain = _flip_gain(adj, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
        if time.time() > deadline:
            break
        if use_two_flips:
            # Try 2-flip moves to escape local optima
            for v in range(n):
                if time.time() > deadline:
                    break
                for w in range(v + 1, n):
                    gain = _two_flip_gain(adj, labels, v, w)
                    if gain > 0:
                        labels[v] = 1 - labels[v]
                        labels[w] = 1 - labels[w]
                        improved = True
                        break
                if improved:
                    break
    return labels


def _simulated_annealing(adj, labels, deadline, start_temp=1.0, cooling=0.999):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp
    
    while time.time() < deadline and temp > 1e-6:
        # Randomly pick a vertex to flip
        v = np.random.randint(n)
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        new_cut = _cut_size(adj, labels)
        delta = new_cut - current_cut
        
        if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-10)):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label
        
        temp *= cooling
    
    return best_labels


def _spectral_init(adj, n):
    """Initialize using the Fiedler vector of the graph Laplacian."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler > 0).astype(np.int8)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        return labels
    except:
        return np.arange(n) % 2


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Set deadline with safety margin
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Phase 1: Multiple restarts with different initializations
    best_labels = None
    best_cut = -1
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 40:  # Reserve 40s for SA and final polish
            break
        
        # Different initialization strategies
        if restart == 0:
            # Spectral initialization
            labels = _spectral_init(adj, n)
        elif restart == 1:
            # Alternating
            labels = np.arange(n) % 2
        elif restart == 2:
            # Random
            labels = np.random.randint(0, 2, n).astype(np.int8)
        elif restart == 3:
            # Random with bias
            labels = (np.random.random(n) > 0.3).astype(np.int8)
        elif restart == 4:
            # Random
            labels = np.random.randint(0, 2, n).astype(np.int8)
        elif restart == 5:
            # Spectral with perturbation
            labels = _spectral_init(adj, n)
            # Perturb some vertices
            for _ in range(5):
                v = np.random.randint(n)
                labels[v] = 1 - labels[v]
        elif restart == 6:
            # Random
            labels = np.random.randint(0, 2, n).astype(np.int8)
        else:
            # Random
            labels = np.random.randint(0, 2, n).astype(np.int8)
        
        # Local search with 2-flip moves for later restarts
        use_2flips = restart >= 3
        labels = _local_search(adj, labels, deadline - 30, use_two_flips=use_2flips)
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 2: Simulated annealing from the best solution
    if time.time() < deadline - 15:
        sa_labels = _simulated_annealing(adj, best_labels, deadline - 10)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Phase 3: Final local search polish with 2-flip moves
    if time.time() < deadline - 2:
        best_labels = _local_search(adj, best_labels, deadline, use_two_flips=True)
    
    return best_labels
# EVOLVE-BLOCK-END
