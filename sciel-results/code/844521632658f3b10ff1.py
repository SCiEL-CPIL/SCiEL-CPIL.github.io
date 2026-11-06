"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)


def _spectral_init(adj, n, k=1):
    """
    Initialize using the (k+1)-th eigenvector of the graph Laplacian.
    Returns a 0/1 label array.
    """
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        if k >= len(eigenvectors):
            k = len(eigenvectors) - 1
        fiedler = eigenvectors[:, k]
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        labels = np.random.randint(0, 2, size=n).astype(np.int8)

    return labels


def _sd_relaxation_rounding(adj, n, num_roundings=5, max_iter=200):
    """
    Use SDP relaxation via projected gradient descent to get a PSD matrix,
    then use randomized hyperplane rounding to generate diverse initializations.
    """
    A = adj.astype(float)
    # Max cut SDP: maximize tr(X^T A) s.t. X is PSD, X_ii = 1
    # We can use a simple approach: start with X = I, then do gradient steps
    # Gradient of tr(X^T A) is A, but we need to project onto PSD with unit diagonal
    
    X = np.eye(n)
    
    for it in range(max_iter):
        # Gradient ascent step
        grad = A.copy()
        # Subtract diagonal to maintain X_ii = 1 constraint (approximately)
        grad[np.diag_indices(n)] = 0
        
        # Simple step size
        step_size = 0.1 / (np.max(np.abs(A)) + 1e-8)
        X = X + step_size * grad
        
        # Project onto PSD with unit diagonal
        # First, set diagonal to 1
        X[np.diag_indices(n)] = 1.0
        
        # Project onto PSD via eigenvalue clipping
        try:
            eigvals, eigvecs = np.linalg.eigh(X)
            eigvals = np.maximum(eigvals, 0)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except Exception:
            break
        
        # Re-enforce unit diagonal after projection
        X[np.diag_indices(n)] = 1.0
    
    # Randomized hyperplane rounding
    initializations = []
    for _ in range(num_roundings):
        # Generate random hyperplane
        u = np.random.randn(n)
        u /= (np.linalg.norm(u) + 1e-8)
        
        # Round based on sign of X * u
        Xu = X @ u
        labels = (Xu > 0).astype(np.int8)
        initializations.append(labels)
    
    return initializations


def _random_init(n):
    """Random 0/1 initialization."""
    return np.random.randint(0, 2, size=n).astype(np.int8)


def _vectorized_greedy_improve(adj, labels, max_passes=50):
    """
    Vectorized greedy local search: repeatedly flip vertices that improve the cut.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()

    for _ in range(max_passes):
        # Vectorized: compute for all vertices at once
        # same_side_mask[v] = 1 if neighbor has same label as v
        same_side_mask = (labels[None, :] == labels[:, None]).astype(np.int8)
        # For each vertex v, count edges to same side and other side
        same_counts = (adj_int * same_side_mask).sum(axis=1)
        other_counts = adj_int.sum(axis=1) - same_counts
        
        # Find vertices where flipping improves the cut
        # Flipping v: gain = same_counts[v] - other_counts[v]
        # If gain > 0, flipping v improves the cut
        gains = same_counts - other_counts
        to_flip = np.where(gains > 0)[0]
        
        if len(to_flip) == 0:
            break
        
        labels[to_flip] = 1 - labels[to_flip]

    return labels


def _goemans_maxwell(adj, labels, max_passes=100):
    """
    Goemans-Williamson style local improvement:
    For each vertex, compute the net gain of flipping it.
    Use a priority queue approach to find the best flip.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)

    for _ in range(max_passes):
        # Compute gains for all vertices
        # gain[v] = (edges to opposite side) - (edges to same side)
        same_counts = (adj_int * (labels[None, :] == labels[:, None]).astype(np.int8)).sum(axis=1)
        other_counts = adj_int.sum(axis=1) - same_counts
        gains = other_counts - same_counts

        # Find vertex with maximum positive gain
        max_gain = gains.max()
        if max_gain <= 0:
            break

        max_gain_idx = np.argmax(gains)
        labels[max_gain_idx] = 1 - labels[max_gain_idx]
        current_cut += max_gain

    return labels


def _simulated_annealing(adj, labels, start_temp=10.0, cool_rate=0.995, max_steps=5000):
    """
    Simple simulated annealing to escape local optima.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp

    for step in range(max_steps):
        v = np.random.randint(0, n)
        old_label = labels[v]
        new_label = 1 - old_label

        # Compute change in cut size
        # Vectorized delta calculation
        # delta = (edges to same side) - (edges to other side)
        # If we flip, same side edges become cut (+1), other side edges become non-cut (-1)
        # So delta = sum(adj[v, same_side]) - sum(adj[v, other_side])
        
        # Efficient calculation:
        # Let S be the set of vertices with label old_label.
        # Let O be the set of vertices with label new_label.
        # v is in S.
        # Edges from v to S (excluding v) become cut.
        # Edges from v to O become non-cut.
        
        # We can precompute degrees or just sum rows.
        # sum(adj[v, labels == old_label]) includes v if self-loop, but graph is simple usually.
        # Assuming no self loops, adj[v,v]=0.
        same_count = np.sum(adj_int[v, labels == old_label])
        other_count = np.sum(adj_int[v, labels != old_label])
        
        delta = same_count - other_count

        new_cut = current_cut + delta

        if new_cut > current_cut:
            accept = True
        else:
            if temp < 1e-10:
                accept = False
            else:
                # Avoid overflow/underflow in exp
                if delta / temp > -50:
                    accept = np.random.random() < np.exp(delta / temp)
                else:
                    accept = False

        if accept:
            labels[v] = new_label
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        temp *= cool_rate

    return best_labels


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Uses spectral initialization + SDP relaxation rounding + vectorized greedy local search + Goemans-Williamson + simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    # Generate multiple initializations
    inits = []

    # Spectral initializations (Fiedler vectors for k=1,2,3)
    for k in range(1, 4):
        spec_labels = _spectral_init(adj, n, k=k)
        inits.append(spec_labels)

    # SDP relaxation with randomized rounding
    if time.time() < deadline - 30:
        sdp_inits = _sd_relaxation_rounding(adj, n, num_roundings=3, max_iter=100)
        inits.extend(sdp_inits)

    # Alternating initialization
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)

    # Several random initializations
    for _ in range(2):
        inits.append(_random_init(n))

    best_labels = None
    best_cut = -1

    for init_labels in inits:
        if time.time() > deadline:
            break

        # Vectorized greedy local search
        greedy_labels = _vectorized_greedy_improve(adj, init_labels)
        greedy_cut = _cut_size(adj, greedy_labels)

        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()

        # Goemans-Williamson refinement (single flip improvements)
        if time.time() < deadline - 10:
            gm_labels = _goemans_maxwell(adj, greedy_labels)
            gm_cut = _cut_size(adj, gm_labels)

            if gm_cut > best_cut:
                best_cut = gm_cut
                best_labels = gm_labels.copy()

        # Simulated annealing from the best result found so far
        if time.time() < deadline - 20:
            sa_labels = _simulated_annealing(adj, best_labels, start_temp=5.0, cool_rate=0.995, max_steps=2000)
            sa_cut = _cut_size(adj, sa_labels)

            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()

    # Final vectorized greedy pass on the best found
    if best_labels is not None and time.time() < deadline:
        final_labels = _vectorized_greedy_improve(adj, best_labels)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()

    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
