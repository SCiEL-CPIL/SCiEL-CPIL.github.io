"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    chosen = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if chosen[v] or forbidden[v]:
            continue
        chosen[v] = True
        count += 1
        # Mark neighbours as forbidden
        forbidden[adjacency[v]] = True
    return count


def _compute_degree(adjacency):
    return adjacency.sum(axis=1).astype(float)


def _compute_neighbor_degree_sum(adjacency):
    # Sum of degrees of neighbours for each vertex
    deg = adjacency.sum(axis=1)
    return (adjacency @ deg).astype(float)


def _compute_coreness(adjacency):
    """Compute k-core number for each vertex using iterative peeling."""
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(int).copy()
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)

    # Bucket vertices by current degree
    max_deg = int(deg.max()) if n > 0 else 0
    # Use a simple approach: repeatedly remove min-degree vertices
    # For 200 nodes, this is fast enough
    active = np.ones(n, dtype=bool)
    current_deg = deg.copy()

    # We'll use a priority-queue-like approach with numpy
    # Since n is small (200), we can do iterative removal
    removed = np.zeros(n, dtype=bool)
    for _ in range(n):
        # Find active vertex with minimum degree
        active_mask = ~removed
        if not active_mask.any():
            break
        active_indices = np.where(active_mask)[0]
        min_deg = current_deg[active_indices].min()
        # Remove one vertex with min degree (first one)
        candidates = active_indices[current_deg[active_indices] == min_deg]
        v = candidates[0]
        removed[v] = True
        core[v] = min_deg
        # Update degrees of neighbours
        for u in np.where(adjacency[v])[0]:
            if not removed[u]:
                current_deg[u] -= 1

    return core.astype(float)


def _search_best_scores(adjacency, time_limit=100.0):
    """
    Search for a good static scoring function by trying various linear combinations
    of degree, neighbor degree sum, and coreness, plus randomized perturbations.
    Returns the best scores array found.
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + time_limit

    # Precompute base features
    degree = _compute_degree(adjacency)
    nds = _compute_neighbor_degree_sum(adjacency)
    coreness = _compute_coreness(adjacency)

    # Normalize features to [0, 1] range for stable combination
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    deg_norm = normalize(degree)
    nds_norm = normalize(nds)
    core_norm = normalize(coreness)

    best_scores = -degree  # Start with the seed strategy (lowest degree first)
    best_count = _greedy_decode(adjacency, best_scores)

    # Try a grid of linear combinations: w1*deg + w2*nds + w3*core
    # Since we want high score = picked first, and low degree is good,
    # we use negative degree or positive "independence potential"
    # Let's define score = a*(-deg_norm) + b*(-nds_norm) + c*core_norm + d*(-deg_norm^2)
    # Actually, let's just try combinations of the raw features with various signs

    # Strategy: try many random linear combinations and also specific known-good ones
    rng = np.random.default_rng(42)

    # First, try some structured combinations
    structured_combos = [
        # (w_deg, w_nds, w_core) applied to (-deg_norm, -nds_norm, core_norm)
        (-1, 0, 0),    # lowest degree
        (0, -1, 0),    # lowest neighbor degree sum
        (0, 0, 1),     # highest coreness
        (-0.5, -0.5, 0),
        (-0.5, 0, 0.5),
        (0, -0.5, 0.5),
        (-0.3, -0.3, 0.4),
        (-0.2, -0.4, 0.4),
        (-0.4, -0.3, 0.3),
        (-1, -1, 0),
        (-1, 0, 1),
        (0, -1, 1),
        (-0.5, -0.5, 0.5),
        (-0.2, -0.2, 0.6),
        (-0.6, -0.2, 0.2),
        (-0.2, -0.6, 0.2),
    ]

    for w1, w2, w3 in structured_combos:
        if time.time() > deadline - 5:
            break
        scores = w1 * (-deg_norm) + w2 * (-nds_norm) + w3 * core_norm
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Now try random perturbations around the best found so far
    # Also try random linear combinations
    num_random_tries = 5000
    for i in range(num_random_tries):
        if time.time() > deadline - 5:
            break
        # Random weights
        w1 = rng.uniform(-1, 0)  # degree: negative means prefer low degree
        w2 = rng.uniform(-1, 0)  # nds: negative means prefer low nds
        w3 = rng.uniform(0, 1)   # core: positive means prefer high core
        scores = w1 * (-deg_norm) + w2 * (-nds_norm) + w3 * core_norm
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Local search: perturb the best scores with small random noise and re-evaluate
    # This helps escape local optima in the discrete greedy decoding space
    local_search_steps = 2000
    current_scores = best_scores.copy()
    current_count = best_count

    for i in range(local_search_steps):
        if time.time() > deadline - 5:
            break
        # Add small random perturbation
        noise = rng.normal(0, 0.05, size=n)
        perturbed = current_scores + noise
        count = _greedy_decode(adjacency, perturbed)
        if count > current_count:
            current_count = count
            current_scores = perturbed.copy()
            if count > best_count:
                best_count = count
                best_scores = perturbed.copy()
        # Occasionally try a larger perturbation
        elif i % 10 == 0:
            noise = rng.normal(0, 0.2, size=n)
            perturbed = current_scores + noise
            count = _greedy_decode(adjacency, perturbed)
            if count > current_count:
                current_count = count
                current_scores = perturbed.copy()
                if count > best_count:
                    best_count = count
                    best_scores = perturbed.copy()

    return best_scores


def priority(adjacency):
    """
    Give every vertex a score. The greedy decoder picks vertices in descending
    score order, skipping neighbours of picked vertices.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([], dtype=float)

    start_time = time.time()
    # Leave a safety margin before the 120s limit
    time_limit = 100.0

    try:
        scores = _search_best_scores(adjacency, time_limit=time_limit)
    except Exception:
        # Fallback to simple degree-based scoring
        scores = -adjacency.sum(axis=1).astype(float)

    return scores
# EVOLVE-BLOCK-END
