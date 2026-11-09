"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gains(adjacency, labels):
    """Compute flip gain for all vertices simultaneously.
    Positive gain means flipping improves the cut.
    """
    n = len(labels)
    # For each vertex v, gain = (# same-side neighbors) - (# opposite-side neighbors)
    # same-side neighbors: adjacency[v] & (labels == labels[v])
    # We can compute this as: for each v, sum(adjacency[v] * (labels == labels[v])) - sum(adjacency[v] * (labels != labels[v]))
    # = 2*sum(adjacency[v] * (labels == labels[v])) - sum(adjacency[v])
    # Using matrix ops: labels as +/-1, then gain[v] = sum(adjacency[v] * labels[v] * labels)
    # Let's use: s = 2*labels - 1 (so labels are +/-1)
    s = 2 * labels - 1  # +1 or -1
    # gain[v] = sum over u of adjacency[v,u] * s[v] * s[u]
    # = s[v] * (adjacency @ s)[v]
    gains = s * (adjacency @ s)
    # Note: this gives 2*(same - other) for each vertex, so positive means flip improves
    return gains


def _local_search(adjacency, labels, max_time=None, use_sa=False, sa_temp=1.0):
    """Local search with optional simulated annealing component."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > max_time:
            break
        
        # Compute all flip gains at once
        gains = _flip_gains(adjacency, labels)
        
        # Find vertices with positive gain
        positive = np.where(gains > 0)[0]
        
        if len(positive) == 0:
            if use_sa and sa_temp > 0.01:
                # Simulated annealing: accept a random flip with some probability
                v = np.random.randint(n)
                if np.random.random() < np.exp(-abs(gains[v]) / sa_temp):
                    labels[v] = 1 - labels[v]
                    current_cut += int(gains[v]) // 2
                    improved = True
                sa_temp *= 0.95  # Cool down
            break
        
        # Flip all vertices with positive gain (parallel update)
        # But to be safe, do sequential in random order
        order = np.random.permutation(positive)
        for v in order:
            if time.time() - start_time > max_time:
                return labels, current_cut
            # Recompute gain for this specific vertex (in case of interactions)
            v_gain = _flip_gains(adjacency, labels)[v]
            if v_gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += int(v_gain) // 2
                improved = True
    
    return labels, current_cut


def _spectral_partition(adjacency):
    """Initialize labels using the Fiedler vector (eigenvector of second smallest eigenvalue)."""
    n = len(adjacency)
    try:
        # Compute eigenvectors of the adjacency matrix
        # Use scipy if available, otherwise numpy
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        A = csr_matrix(adjacency)
        # Find the Fiedler vector (eigenvector corresponding to second smallest eigenvalue)
        # For max cut, we want to use the eigenvector of the Laplacian
        # L = D - A
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        L = np.diag(degrees) - adjacency
        # Find smallest eigenvalues of L
        vals, vecs = eigsh(L, k=2, which='SM')
        fiedler = vecs[:, 1]  # Second smallest eigenvector
        # Assign labels based on sign
        labels = (fiedler > 0).astype(int)
        # Ensure balanced partition
        if labels.sum() > n / 2:
            labels = 1 - labels
        return labels
    except Exception:
        # Fallback: use numpy eig
        try:
            degrees = np.array(adjacency.sum(axis=1)).flatten()
            L = np.diag(degrees) - adjacency
            vals, vecs = np.linalg.eigh(L)
            fiedler = vecs[:, 1]
            labels = (fiedler > 0).astype(int)
            if labels.sum() > n / 2:
                labels = 1 - labels
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def _try_labels(labels, time_limit, use_sa=False):
        nonlocal best_labels, best_cut
        labels, cut = _local_search(adjacency, labels, max_time=time_limit, use_sa=use_sa)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        return cut
    
    # Phase 1: Spectral initialization
    remaining = deadline - time.time()
    if remaining > 20:
        spectral_labels = _spectral_partition(adjacency)
        _try_labels(spectral_labels, min(remaining - 5, 30))
    
    # Phase 2: Multiple random restarts with local search + SA
    num_restarts = 20
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 10:
            break
        
        # Choose initialization strategy
        if best_labels is not None and restart % 3 == 0:
            # Perturbed best solution
            current_labels = best_labels.copy()
            num_flips = max(1, int(n * 0.1))
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            current_labels[flip_indices] = 1 - current_labels[flip_indices]
        elif restart % 3 == 1:
            # Random start
            current_labels = np.random.randint(0, 2, size=n)
        else:
            # Alternating start with random perturbation
            current_labels = np.arange(n) % 2
            if np.random.random() < 0.5:
                current_labels = 1 - current_labels
        
        # Time budget for this restart
        time_per_restart = max(2, (remaining - 5) / max(1, num_restarts - restart))
        use_sa = (restart % 2 == 0)  # Use SA on even restarts
        _try_labels(current_labels, time_per_restart, use_sa)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
