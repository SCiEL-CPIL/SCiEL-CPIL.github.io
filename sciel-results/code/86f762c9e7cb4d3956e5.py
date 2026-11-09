"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(A, labels):
    """Number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((A & diff).sum() // 2)


def _spectral_start(A, n, rng):
    """Start from the sign of the Fiedler (2nd) eigenvector, with random fallback."""
    try:
        w, V = np.linalg.eigh(A)
        # eigh returns ascending eigenvalues; Fiedler vector is the second.
        fiedler = V[:, 1]
        labels = (fiedler >= 0).astype(np.int64)
        # Ensure both labels are present; if degenerate, fall back to random.
        if labels.min() == labels.max():
            labels = rng.integers(0, 2, size=n)
        return labels
    except Exception:
        return rng.integers(0, 2, size=n)


def _local_search(A, labels, max_sweeps=200):
    """Vectorized greedy local improvement. Returns (labels, cut)."""
    n = len(labels)
    row_sums = A.sum(axis=1)
    labels = labels.copy()

    for _ in range(max_sweeps):
        # sum of labels over neighbors of each vertex
        A_labels = A @ labels
        # For vertex v: edges to opposite set = A_labels[v] - (2*labels[v]-1)*row_sums[v]/... 
        # Simpler: let s = labels (0/1). For v, same-set neighbors = A_labels[v] - (2*labels[v]-1)*row_sums[v] ... 
        # Use direct formula:
        # cut contribution of v to opposite set = A_labels[v] if labels[v]==0 else row_sums[v]-A_labels[v]
        # gain of flipping v = (edges to same set) - (edges to opposite set)
        # edges to opposite = A_labels[v] if labels[v]==0 else row_sums[v]-A_labels[v]
        # edges to same     = row_sums[v]-A_labels[v] if labels[v]==0 else A_labels[v]
        opp = np.where(labels == 0, A_labels, row_sums - A_labels)
        same = row_sums - opp
        gain = same - opp
        flip = gain > 0
        if not flip.any():
            break
        labels[flip] = 1 - labels[flip]

    cut = _cut_size(A, labels)
    return labels, cut


def _polish(A, labels, max_sweeps=1000):
    """Tight final polish with more sweeps."""
    return _local_search(A, labels, max_sweeps=max_sweeps)


def partition(adjacency):
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin under 120s

    A = np.asarray(adjacency, dtype=np.int64)
    n = A.shape[0]
    rng = np.random.default_rng(12345)

    best_labels = None
    best_cut = -1

    # Initial spectral-based start.
    labels = _spectral_start(A, n, rng)
    labels, cut = _local_search(A, labels, max_sweeps=100)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()

    # Multiple random restarts with local search, bounded by the deadline.
    restarts = 0
    while time.time() < DEADLINE - 5.0:
        labels = rng.integers(0, 2, size=n)
        labels, cut = _local_search(A, labels, max_sweeps=60)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        restarts += 1
        # If we've done many restarts and are improving slowly, tighten.
        if restarts % 4 == 0 and time.time() > DEADLINE - 15.0:
            break

    # Final high-precision polish on the best solution found.
    if time.time() < DEADLINE - 2.0:
        labels, cut = _polish(A, best_labels, max_sweeps=2000)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Guarantee a valid 0/1 label vector.
    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int64)
    return best_labels
# EVOLVE-BLOCK-END
