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


def _scores_neighbor_value(adjacency):
    """
    Score based on the sum of 'values' of neighbors.
    Low-degree neighbors are 'valuable' for MIS (easier to place).
    If a vertex blocks many valuable neighbors, it should have LOW priority.
    value(u) = 1 / (1 + degree(u))
    score(v) = - sum(value(u) for u in N(v))
    """
    d = _degree(adjacency)
    values = 1.0 / (1.0 + d)
    return -(adjacency @ values)


def _scores_local_density(adjacency):
    """
    Score based on sum of neighbor degrees.
    Vertices connected to high-degree vertices are in 'dense' parts.
    Generally, lower local density is better for MIS?
    """
    d = _degree(adjacency)
    return -(adjacency @ d)


def _local_search(adjacency, init_scores, rng, time_limit):
    """
    Hill climbing on the score vector to maximize greedy independent set size.
    """
    n = adjacency.shape[0]
    best_scores = init_scores.copy()
    order = np.argsort(-best_scores)
    best_size = _greedy_size(adjacency, order)
    
    current_scores = init_scores.copy()
    start_time = time.time()
    timeout = start_time + time_limit
    noise_scale = 0.5
    
    while time.time() < timeout:
        # Perturb scores
        noise = rng.normal(0, noise_scale, n)
        perturbed = current_scores + noise
        
        order = np.argsort(-perturbed)
        size = _greedy_size(adjacency, order)
        
        if size > best_size:
            best_size = size
            best_scores = perturbed.copy()
            current_scores = perturbed.copy()
            # Reduce noise slightly on success? Or keep it.
            noise_scale = max(0.1, noise_scale * 0.99)
        elif size == best_size:
            # Accept with small probability to escape plateaus
            if rng.random() < 0.05:
                current_scores = perturbed.copy()
        else:
            # Reject
            pass
            
        # Occasionally restart from best to avoid local minima
        if rng.random() < 0.01:
            current_scores = best_scores.copy()
            noise_scale = 0.5

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
    candidates.append(_scores_neighbor_value(adjacency))
    candidates.append(_scores_local_density(adjacency))
    candidates.append(_scores_random(adjacency, rng))
    
    # Combine some features
    deg = _degree(adjacency).astype(float)
    nv = _scores_neighbor_value(adjacency)
    candidates.append(0.5 * _scores_low_degree(adjacency) + 0.5 * nv)

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
    if time_left() > 5:
        # Use remaining time for local search
        search_time = time_left() - 2.0 # Leave 2s margin
        if search_time > 0:
            best_scores = _local_search(adjacency, best_scores, rng, search_time)

    # Ensure higher score = selected first
    return best_scores
# EVOLVE-BLOCK-END
