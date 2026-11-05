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


def _simulated_annealing(labels, adj, deadline, rng):
    """
    Simulated annealing on the cut problem.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_labels = labels.copy()
    best_cut = current_cut

    T_start = 10.0
    T_end = 0.01
    n_iter = 3000

    for it in range(n_iter):
        if time.time() > deadline:
            break

        if it < n_iter - 1:
            T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
        else:
            T = T_end

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
    For MaxCut, we solve the SDP relaxation approximately.
    Since we don't have an SDP solver, we use a heuristic:
    Use the eigenvector of the Laplacian or adjacency matrix as a proxy for the SDP solution.
    """
    n = adj.shape[0]
    # Use the Fiedler vector (2nd smallest eigenvector of Laplacian) as a starting point
    # This is a common heuristic for graph partitioning
    try:
        L = csgraph_laplacian(adj, normed=False)
        # Get the 2nd smallest eigenvector
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        # Round to 0/1 based on sign
        labels = (fiedler >= 0).astype(int)
    except Exception:
        # Fallback to random
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _ilp_polish(labels, adj, deadline):
    """
    Final high-precision polish: try all pairs of flips to escape 2-flip local optima.
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

    # Then try double-flip moves
    if time.time() > deadline - 1.0:
        return best_labels, best_cut
    
    for _ in range(5):  # up to 5 rounds of double-flip
        if time.time() > deadline - 0.5:
            break
        best_double_gain = 0
        best_pair = None
        gains = _vectorized_gains(labels, adj)
        
        # Sample some pairs to try
        rng = np.random.default_rng(42)
        for _ in range(1000):
            if time.time() > deadline - 0.5:
                break
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            # Compute gain of flipping both i and j
            # This is approximate; exact would require more computation
            # For simplicity, we'll just try it and measure
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            new_cut = _cut_size(labels, adj)
            delta = new_cut - current_cut
            if delta > best_double_gain:
                best_double_gain = delta
                best_pair = (i, j)
            # Revert
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
        
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


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses multiple strategies: Fiedler vector initialization, simulated annealing,
    greedy polish, and double-flip local search.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(42)

    # Phase 1: Fiedler vector initialization + greedy polish
    if time.time() < deadline - 10.0:
        labels_fiedler = _goemans_maxwfn(adj, deadline)
        labels_fiedler, cut_fiedler = _greedy_polish(labels_fiedler, adj, deadline)
        best_labels = labels_fiedler.copy()
        best_cut = cut_fiedler

    # Phase 2: Alternating start + greedy polish
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _greedy_polish(labels_alt, adj, deadline)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()

    # Phase 3: Random restarts with simulated annealing
    n_restarts_max = 30
    for r in range(n_restarts_max):
        if time.time() > deadline - 10.0:
            break

        # Random initial labeling
        labels = rng.integers(0, 2, size=n)
        labels, cut = _simulated_annealing(labels, adj, deadline, rng)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final: high-precision polish on the best found
    if time.time() < deadline - 2.0:
        best_labels, best_cut = _ilp_polish(best_labels, adj, deadline)

    return best_labels
# EVOLVE-BLOCK-END
