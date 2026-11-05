"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import scipy.sparse.linalg as spla

DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))


def _vectorized_gains(labels, adj):
    """
    Compute flip gain for every vertex in a vectorized way.
    gain[v] = change in cut size if we flip vertex v.
    """
    deg = adj.sum(axis=1)
    neighbor_label_sum = adj @ labels
    sign = 2 * labels - 1
    gains = sign * (2 * neighbor_label_sum - deg)
    return gains


def _greedy_polish(labels, adj, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_cut = current_cut
    best_labels = labels.copy()

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        gains = _vectorized_gains(labels, adj)
        max_gain = int(gains.max())
        if max_gain > 0:
            best_v = int(np.argmax(gains))
            labels[best_v] = 1 - labels[best_v]
            current_cut += max_gain
            improved = True
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

    return best_labels, best_cut


def _simulated_annealing(labels, adj, deadline, rng, n_iter=5000):
    """
    Simulated annealing on the cut problem with spin-glass interpretation.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_labels = labels.copy()
    best_cut = current_cut

    T_start = 5.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(n_iter - 1, 1))

    for it in range(n_iter):
        if time.time() > deadline:
            break

        T = T_start * (cooling ** it)

        gains = _vectorized_gains(labels, adj)

        v = rng.integers(0, n)
        delta = gains[v]

        if delta >= 0:
            labels[v] = 1 - labels[v]
            current_cut += int(delta)
        else:
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += int(delta)

        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()

    return best_labels, best_cut


def _goemans_maxwfn(adj, deadline):
    """
    Goemans-Williamson style rounding via random hyperplane.
    Uses the Fiedler vector as a proxy for the SDP solution.
    """
    n = adj.shape[0]
    try:
        L = csgraph_laplacian(adj, normed=False)
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        labels = (fiedler >= 0).astype(int)
    except Exception:
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _double_flip_polish(labels, adj, deadline):
    """
    Final high-precision polish: try pairs of flips to escape 2-flip local optima.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_cut = current_cut
    best_labels = labels.copy()

    # First do single-flip greedy
    improved = True
    while improved:
        if time.time() > deadline:
            return best_labels, best_cut
        improved = False
        gains = _vectorized_gains(labels, adj)
        max_gain = int(gains.max())
        if max_gain > 0:
            best_v = int(np.argmax(gains))
            labels[best_v] = 1 - labels[best_v]
            current_cut += max_gain
            improved = True
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

    # Then try double-flip moves with sampling
    if time.time() > deadline - 1.0:
        return best_labels, best_cut
    
    rng = np.random.default_rng(42)
    for _ in range(10):  # up to 10 rounds of double-flip
        if time.time() > deadline - 0.5:
            break
        best_double_gain = 0
        best_pair = None
        gains = _vectorized_gains(labels, adj)
        
        # Sample pairs to try
        n_samples = min(2000, n * (n - 1) // 2)
        for _ in range(n_samples):
            if time.time() > deadline - 0.5:
                break
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            
            # Compute exact gain of flipping both i and j
            # Flip i: gain_i
            # Flip j: gain_j
            # But if i and j are connected, there's an interaction term
            # gain(i,j) = gain_i + gain_j + 2*adj[i,j]*(2*labels[i]*labels[j] - 1)
            # Actually, let's just compute it directly for correctness
            old_i, old_j = labels[i], labels[j]
            labels[i] = 1 - old_i
            labels[j] = 1 - old_j
            new_cut = _cut_size(labels, adj)
            delta = new_cut - current_cut
            if delta > best_double_gain:
                best_double_gain = delta
                best_pair = (i, j)
            labels[i] = old_i
            labels[j] = old_j
        
        if best_pair is not None and best_double_gain > 0:
            i, j = best_pair
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            current_cut += best_double_gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            break

    return best_labels, best_cut


def _spectral_init(adj, n, rng):
    """
    Generate multiple spectral initializations using different eigenvectors.
    """
    inits = []
    
    # Fiedler vector
    try:
        L = csgraph_laplacian(adj, normed=False)
        vals, vecs = spla.eigsh(L, k=min(4, n - 1), which='SM')
        idx = np.argsort(vals)
        for k in range(min(3, len(idx))):
            vec = vecs[:, idx[k]]
            labels = (vec >= 0).astype(int)
            inits.append(labels)
    except Exception:
        pass
    
    # Random hyperplane from adjacency eigenvector
    try:
        vals, vecs = spla.eigsh(adj, k=min(4, n - 1), which='LM')
        idx = np.argsort(vals)
        for k in range(min(3, len(idx))):
            vec = vecs[:, idx[k]]
            labels = (vec >= 0).astype(int)
            inits.append(labels)
    except Exception:
        pass
    
    # Alternating
    inits.append(np.arange(n) % 2)
    
    # Random
    inits.append(rng.integers(0, 2, size=n))
    
    return inits


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses spectral initialization, simulated annealing, and multi-stage local search.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(42)

    # Generate diverse initializations
    inits = _spectral_init(adj, n, rng)

    # Phase 1: Polish each initialization with greedy
    for labels_init in inits:
        if time.time() > deadline - 20.0:
            break
        labels_p, cut_p = _greedy_polish(labels_init, adj, deadline)
        if cut_p > best_cut:
            best_cut = cut_p
            best_labels = labels_p.copy()

    # Phase 2: Simulated annealing from multiple starts
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break

        # Use a different initialization for each restart
        if r < len(inits):
            labels = inits[r].copy()
        else:
            labels = rng.integers(0, 2, size=n)

        # Adaptive iteration count based on remaining time
        remaining = deadline - time.time()
        n_iter = int(min(5000, max(1000, remaining * 30)))
        
        labels_sa, cut_sa = _simulated_annealing(labels, adj, deadline, rng, n_iter=n_iter)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()

    # Phase 3: More SA restarts if time permits
    for r in range(5):
        if time.time() > deadline - 10.0:
            break
        labels = rng.integers(0, 2, size=n)
        n_iter = int(min(3000, max(500, (deadline - time.time()) * 20)))
        labels_sa, cut_sa = _simulated_annealing(labels, adj, deadline, rng, n_iter=n_iter)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa.copy()

    # Final: double-flip polish on the best found
    if time.time() < deadline - 2.0 and best_labels is not None:
        best_labels, best_cut = _double_flip_polish(best_labels, adj, deadline)

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)

    return best_labels
# EVOLVE-BLOCK-END
