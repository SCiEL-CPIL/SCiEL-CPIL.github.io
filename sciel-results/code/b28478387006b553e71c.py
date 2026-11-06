"""Maximum Cut via spectral + SDP initialization, greedy local search, and simulated annealing."""
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


def _sd_relaxation_rounding(adj, n, num_roundings=5, max_iter=150):
    """
    Use SDP relaxation via projected gradient descent to get a PSD matrix,
    then use randomized hyperplane rounding to generate diverse initializations.
    """
    A = adj.astype(float)
    X = np.eye(n)

    for it in range(max_iter):
        grad = A.copy()
        grad[np.diag_indices(n)] = 0

        step_size = 0.1 / (np.max(np.abs(A)) + 1e-8)
        X = X + step_size * grad

        X[np.diag_indices(n)] = 1.0

        try:
            eigvals, eigvecs = np.linalg.eigh(X)
            eigvals = np.maximum(eigvals, 0)
            X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except Exception:
            break

        X[np.diag_indices(n)] = 1.0

    initializations = []
    for _ in range(num_roundings):
        u = np.random.randn(n)
        u /= (np.linalg.norm(u) + 1e-8)
        Xu = X @ u
        labels = (Xu > 0).astype(np.int8)
        initializations.append(labels)

    return initializations


def _random_init(n):
    """Random 0/1 initialization."""
    return np.random.randint(0, 2, size=n).astype(np.int8)


def _vectorized_greedy_improve(adj, labels, max_passes=100):
    """
    Vectorized greedy local search: repeatedly flip vertices that improve the cut.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()

    for _ in range(max_passes):
        same_side_mask = (labels[None, :] == labels[:, None]).astype(np.int8)
        same_counts = (adj_int * same_side_mask).sum(axis=1)
        other_counts = adj_int.sum(axis=1) - same_counts

        gains = same_counts - other_counts
        to_flip = np.where(gains > 0)[0]

        if len(to_flip) == 0:
            break

        labels[to_flip] = 1 - labels[to_flip]

    return labels


def _goemans_maxwell(adj, labels, max_passes=200):
    """
    Goemans-Williamson style local improvement:
    For each vertex, compute the net gain of flipping it.
    Use a priority queue approach to find the best flip.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()

    for _ in range(max_passes):
        same_counts = (adj_int * (labels[None, :] == labels[:, None]).astype(np.int8)).sum(axis=1)
        other_counts = adj_int.sum(axis=1) - same_counts
        gains = other_counts - same_counts

        max_gain = gains.max()
        if max_gain <= 0:
            break

        max_gain_idx = np.argmax(gains)
        labels[max_gain_idx] = 1 - labels[max_gain_idx]

    return labels


def _simulated_annealing(adj, labels, start_temp=10.0, cool_rate=0.997, max_steps=8000):
    """
    Simulated annealing to escape local optima.
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


def _pairwise_swap_search(adj, labels, max_passes=50):
    """
    Try swapping pairs of vertices to improve the cut.
    This can escape some local optima that single-flip search cannot.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)

    for _ in range(max_passes):
        improved = False
        # Sample random pairs to try swapping
        for _ in range(200):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue

            # Compute delta for swapping i and j
            # If labels[i] != labels[j], swapping changes both
            # If labels[i] == labels[j], swapping does nothing

            if labels[i] == labels[j]:
                continue

            # Compute gain from swapping i and j
            # For vertex i: currently has label L_i, will get label L_j
            # For vertex j: currently has label L_j, will get label L_i
            # Net change = (gain for i) + (gain for j) - (interaction term)

            # Simpler: just compute cut before and after for the affected vertices
            # But that's expensive. Let's use a formula.

            # Let's just try the swap and check if it improves
            labels[i], labels[j] = labels[j], labels[i]
            new_cut = _cut_size(adj, labels)
            if new_cut > current_cut:
                current_cut = new_cut
                improved = True
            else:
                labels[i], labels[j] = labels[j], labels[i]

        if not improved:
            break

    return labels


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Uses spectral initialization + SDP relaxation rounding + vectorized greedy local search
    + Goemans-Williamson + simulated annealing + pairwise swap search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    # Generate multiple initializations
    inits = []

    # Spectral initializations (Fiedler vectors for k=1,2,3)
    for k in range(1, 4):
        spec_labels = _spectral_init(adj, n, k=k)
        inits.append(spec_labels)

    # SDP relaxation with randomized rounding
    if time.time() < deadline - 40:
        sdp_inits = _sd_relaxation_rounding(adj, n, num_roundings=5, max_iter=150)
        inits.extend(sdp_inits)

    # Alternating initialization
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)

    # Several random initializations
    for _ in range(3):
        inits.append(_random_init(n))

    best_labels = None
    best_cut = -1

    for init_labels in inits:
        if time.time() > deadline:
            break

        # Vectorized greedy local search
        greedy_labels = _vectorized_greedy_improve(adj, init_labels, max_passes=100)
        greedy_cut = _cut_size(adj, greedy_labels)

        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()

        # Goemans-Williamson refinement (single flip improvements)
        if time.time() < deadline - 15:
            gm_labels = _goemans_maxwell(adj, greedy_labels, max_passes=200)
            gm_cut = _cut_size(adj, gm_labels)

            if gm_cut > best_cut:
                best_cut = gm_cut
                best_labels = gm_labels.copy()

        # Pairwise swap search
        if time.time() < deadline - 25:
            swap_labels = _pairwise_swap_search(adj, best_labels, max_passes=30)
            swap_cut = _cut_size(adj, swap_labels)

            if swap_cut > best_cut:
                best_cut = swap_cut
                best_labels = swap_labels.copy()

        # Simulated annealing from the best result found so far
        if time.time() < deadline - 30:
            sa_labels = _simulated_annealing(adj, best_labels, start_temp=8.0, cool_rate=0.997, max_steps=6000)
            sa_cut = _cut_size(adj, sa_labels)

            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()

    # Final refinement passes
    if best_labels is not None and time.time() < deadline - 5:
        # Final vectorized greedy pass
        final_labels = _vectorized_greedy_improve(adj, best_labels, max_passes=100)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()

        # Final Goemans-Williamson pass
        if time.time() < deadline - 5:
            gm_final = _goemans_maxwell(adj, best_labels, max_passes=200)
            gm_final_cut = _cut_size(adj, gm_final)
            if gm_final_cut > best_cut:
                best_cut = gm_final_cut
                best_labels = gm_final.copy()

        # One more simulated annealing pass with lower temperature
        if time.time() < deadline - 5:
            sa_final = _simulated_annealing(adj, best_labels, start_temp=3.0, cool_rate=0.998, max_steps=3000)
            sa_final_cut = _cut_size(adj, sa_final)
            if sa_final_cut > best_cut:
                best_cut = sa_final_cut
                best_labels = sa_final.copy()

    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
