"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 120s hard limit


def _greedy_size(adjacency, order):
    """Return the size of the independent set produced by a fixed greedy decoder."""
    n = len(order)
    selected = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not selected[v]:
            selected[v] = True
            count += 1
            # Mark neighbours as skipped (vectorized)
            selected[adjacency[v]] = True
    return count


def _degree(adjacency):
    return adjacency.sum(axis=1)


def _scores_low_degree(adjacency):
    """Seed: lowest degree first (negative degree)."""
    return -_degree(adjacency).astype(float)


def _scores_random(adjacency, rng):
    return rng.random(adjacency.shape[0])


def _scores_pagerank_like(adjacency, max_iter=50, tol=1e-10, alpha=0.85):
    """A PageRank-like centrality; high centrality first tends to block many, so we invert."""
    n = adjacency.shape[0]
    d = _degree(adjacency)
    # Avoid division by zero
    inv_d = np.where(d > 0, 1.0 / d, 0.0)
    r = np.ones(n) / n
    for _ in range(max_iter):
        new_r = (1 - alpha) / n + alpha * (adjacency.T @ (r * inv_d))
        diff = np.sum(np.abs(new_r - r))
        r = new_r
        if diff < tol:
            break
    # We want low-centrality vertices first to maximize independent set size
    return -r


def _scores_clustering(adjacency):
    """Score based on local clustering coefficient; low clustering first."""
    n = adjacency.shape[0]
    d = _degree(adjacency)
    # For each vertex, count triangles via A^3 diagonal
    A3 = adjacency @ adjacency @ adjacency
    tri_diag = np.diag(A3)
    # Clustering coefficient c_i = 2*tri_i / (d_i*(d_i-1))
    denom = d * (d - 1)
    c = np.where(denom > 0, 2 * tri_diag / denom, 0.0)
    return -c


def _scores_coreness(adjacency):
    """Approximate k-core number; low coreness first."""
    n = adjacency.shape[0]
    d = _degree(adjacency).copy()
    # Simple iterative peeling
    core = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    while True:
        changed = False
        for v in range(n):
            if active[v] and d[v] < np.max(core[active]) if np.any(active) else False:
                pass
        # Simplified: use degree as proxy for coreness in ER graphs
        break
    return -d.astype(float)


def _simulate_and_refine(adjacency, base_scores, rng, max_simulations=200):
    """
    Refine scores by simulating the greedy decoder and adjusting scores
    based on which vertices end up selected vs skipped.
    """
    n = adjacency.shape[0]
    scores = base_scores.copy()
    best_size = -1
    best_scores = scores.copy()

    # Initial simulation
    order = np.argsort(-scores)
    size = _greedy_size(adjacency, order)
    best_size = size
    best_scores = scores.copy()

    for _ in range(max_simulations):
        # Perturb scores
        noise = rng.normal(0, 0.1, n)
        perturbed = scores + noise
        order = np.argsort(-perturbed)
        size = _greedy_size(adjacency, order)
        if size > best_size:
            best_size = size
            best_scores = perturbed.copy()
            scores = perturbed.copy()
        # Update scores: boost vertices that were selected, penalize skipped
        selected_mask = np.zeros(n, dtype=bool)
        for v in order:
            if not selected_mask[v]:
                selected_mask[v] = True
                selected_mask[adjacency[v]] = True
        # Vertices that were selected get a small boost; skipped get a penalty
        scores = best_scores + 0.05 * (selected_mask.astype(float) - 0.5)
        scores = np.clip(scores, -100, 100)

    return best_scores


def priority(adjacency):
    """
    Priority function for Maximum Independent Set via greedy decoder.
    Returns one score per vertex; higher score = selected earlier.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    rng = np.random.default_rng(42)

    # Deadline guard
    def time_left():
        return _START_TIME + 120.0 - _DEADLINE_MARGIN - time.time()

    # Generate candidate score vectors
    candidates = []
    candidates.append(_scores_low_degree(adjacency))
    candidates.append(_scores_pagerank_like(adjacency))
    candidates.append(_scores_clustering(adjacency))
    candidates.append(_scores_random(adjacency, rng))

    # Evaluate each candidate
    best_size = -1
    best_scores = candidates[0]
    for scores in candidates:
        if time_left() < 5:
            break
        order = np.argsort(-scores)
        size = _greedy_size(adjacency, order)
        if size > best_size:
            best_size = size
            best_scores = scores

    # Refine the best candidate with local search
    if time_left() > 10:
        best_scores = _simulate_and_refine(adjacency, best_scores, rng,
                                           max_simulations=int(min(100, time_left() / 0.1)))

    # Ensure higher score = selected first
    return best_scores
# EVOLVE-BLOCK-END
