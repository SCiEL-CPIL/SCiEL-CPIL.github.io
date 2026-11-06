"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Number of edges crossing the partition, O(n^2) via matrix identity."""
    # cut = 1/4 * sum_i sum_j adj[i,j] * (1 - 2*l_i) * (1 - 2*l_j)
    #     = 1/4 * s @ (2*A - 1) @ s,  where s = 1 - 2*labels
    s = 1 - 2 * labels.astype(np.int32)
    val = s @ (2 * adj - 1) @ s
    return int(val // 4)


def _gains(adj, labels, deg):
    """
    Flip gain for each vertex (vectorized).
    gain[v] = 2*(edges from v to opposite side) - deg[v].
    Positive gain means flipping v increases the cut.
    """
    s = 1 - 2 * labels.astype(np.int32)
    return 2 * (adj.astype(np.int32) @ s) - deg


def _local_search(adj, labels, deg, time_limit=None):
    """
    Fast greedy local search: repeatedly flip the vertex with max positive gain.
    Vectorized gain computation per sweep.
    """
    labels = labels.copy()
    cur = _cut_size(adj, labels)
    gains = _gains(adj, labels, deg)
    n = len(labels)

    while True:
        idx = int(np.argmax(gains))
        g = gains[idx]
        if g <= 0:
            break
        # flip idx
        labels[idx] = 1 - labels[idx]
        cur += g
        # update gains for idx and its neighbors
        nbrs = np.nonzero(adj[idx])[0]
        sub = np.concatenate(([idx], nbrs))
        gains[sub] = _gains(adj, labels, deg)[sub]
        if time_limit is not None and time.time() > time_limit:
            break

    return labels, cur


def _spectral_init(adj, n):
    """Fiedler-vector (second eigenvector of Laplacian) based 0/1 labels."""
    try:
        deg = adj.sum(axis=1).astype(float)
        L = np.diag(deg) - adj.astype(float)
        w, V = np.linalg.eigh(L)
        fiedler = V[:, 1]
        # threshold at the median for a balanced partition
        med = np.median(fiedler)
        labels = (fiedler > med).astype(np.int8)
        if labels.sum() == 0 or labels.sum() == n:
            labels = (fiedler > 0).astype(np.int8)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _simulated_annealing(adj, labels, deg, cur, time_limit):
    """
    Simulated annealing over single-vertex flips, vectorized gain lookup.
    Returns (best_labels, best_cut).
    """
    labels = labels.copy()
    best_labels = labels.copy()
    best_cut = cur
    temp = 8.0
    cool = 0.9995
    n = len(labels)
    rng = np.random

    while time.time() < time_limit:
        if temp < 1e-6:
            break
        v = rng.randint(0, n)
        g = _gains(adj, labels, deg)[v]
        if g > 0:
            accept = True
        else:
            accept = rng.random() < np.exp(g / temp)
        if accept:
            labels[v] = 1 - labels[v]
            cur += g
            if cur > best_cut:
                best_cut = cur
                best_labels = labels.copy()
        temp *= cool

    return best_labels, best_cut


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Spectral + random + alternating restarts, each refined by fast local search,
    followed by simulated annealing on the best.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before the 150s kill

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(np.int32)
    adj_int = adj.astype(np.int32)

    # Build a diverse pool of initializations
    inits = []
    inits.append(_spectral_init(adj, n))
    inits.append(np.arange(n) % 2)
    # degree-based: high degree on one side
    order = np.argsort(-deg)
    db = np.zeros(n, dtype=np.int8)
    db[order[: n // 2]] = 1
    inits.append(db)
    for _ in range(12):
        inits.append(np.random.randint(0, 2, size=n).astype(np.int8))

    best_labels = None
    best_cut = -1

    # Local search from each initialization
    for init in inits:
        if time.time() > deadline:
            break
        ls_labels, ls_cut = _local_search(adj_int, init, deg, time_limit=deadline)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()

    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj_int, best_labels)

    # Simulated annealing from the best found
    if time.time() < deadline:
        sa_labels, sa_cut = _simulated_annealing(adj_int, best_labels, deg, best_cut,
                                                 time_limit=deadline - 5)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    # Final local search polish on the best
    if time.time() < deadline:
        fin_labels, fin_cut = _local_search(adj_int, best_labels, deg, time_limit=deadline)
        if fin_cut > best_cut:
            best_cut = fin_cut
            best_labels = fin_labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
