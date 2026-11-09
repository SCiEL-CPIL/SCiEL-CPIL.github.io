"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_is_size(adjacency, scores):
    """Simulate greedy decoder: pick vertices in descending score order, skip neighbors of picked."""
    n = len(scores)
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not picked[v]:
            # Mark v and all its neighbors as picked (unavailable)
            picked[v] = True
            picked[adjacency[v]] = True
            count += 1
    return count


def _score_degree_low(adjacency):
    """Lowest degree first."""
    return -adjacency.sum(axis=1).astype(float)


def _score_degree_high(adjacency):
    """Highest degree first."""
    return adjacency.sum(axis=1).astype(float)


def _score_neighbor_degree(adjacency):
    """Sum of degrees of neighbors (prefer vertices connected to high-degree nodes)."""
    deg = adjacency.sum(axis=1)
    return -(adjacency @ deg).astype(float)


def _score_clustering(adjacency):
    """Approximate clustering coefficient: edges among neighbors / possible edges."""
    n = adjacency.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        k = len(neighbors)
        if k < 2:
            scores[i] = 0.0
        else:
            sub = adjacency[np.ix_(neighbors, neighbors)]
            edges = sub.sum() / 2.0
            scores[i] = -edges / (k * (k - 1) / 2.0)
    return scores


def _score_core_number(adjacency):
    """Approximate k-core number via iterative peeling."""
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float).copy()
    core = np.zeros(n)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        min_deg = deg[remaining].min()
        mask = remaining & (deg <= min_deg)
        core[mask] = min_deg
        remaining[~mask] = False
        # Remove those nodes
        for v in np.where(mask)[0]:
            nbrs = np.where(adjacency[v] & remaining)[0]
            deg[nbrs] -= 1
    return -core


def _score_local_density(adjacency):
    """Negative local density: prefer vertices in sparser neighborhoods."""
    n = adjacency.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        k = len(neighbors)
        if k < 2:
            scores[i] = 0.0
        else:
            sub = adjacency[np.ix_(neighbors, neighbors)]
            edges = sub.sum() / 2.0
            density = edges / (k * (k - 1) / 2.0)
            scores[i] = -density
    return scores


def _score_degree_squared(adjacency):
    """Negative square of degree."""
    deg = adjacency.sum(axis=1)
    return -(deg ** 2).astype(float)


def _score_harmonic_degree(adjacency):
    """1/(1+degree) - prefer low degree with saturation."""
    deg = adjacency.sum(axis=1).astype(float)
    return 1.0 / (1.0 + deg)


def priority(adjacency):
    """
    Adaptively select the best scoring strategy by simulating the greedy decoder
    on the actual input graph and picking the strategy that yields the largest
    independent set.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    # Define candidate scoring functions
    candidates = [
        ("degree_low", _score_degree_low),
        ("degree_high", _score_degree_high),
        ("neighbor_degree", _score_neighbor_degree),
        ("clustering", _score_clustering),
        ("core_number", _score_core_number),
        ("local_density", _score_local_density),
        ("degree_squared", _score_degree_squared),
        ("harmonic_degree", _score_harmonic_degree),
    ]

    best_size = -1
    best_scores = None

    for name, func in candidates:
        if time.time() > deadline:
            break
        try:
            scores = func(adjacency)
            size = _greedy_is_size(adjacency, scores)
            if size > best_size:
                best_size = size
                best_scores = scores
        except Exception:
            continue

    # Fallback if nothing worked
    if best_scores is None:
        best_scores = _score_degree_low(adjacency)

    return best_scores
# EVOLVE-BLOCK-END
