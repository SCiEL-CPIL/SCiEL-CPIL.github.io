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
    removed = np.zeros(n, dtype=bool)
    current_deg = deg.copy()

    for _ in range(n):
        active_mask = ~removed
        if not active_mask.any():
            break
        active_indices = np.where(active_mask)[0]
        min_deg = current_deg[active_indices].min()
        candidates = active_indices[current_deg[active_indices] == min_deg]
        v = candidates[0]
        removed[v] = True
        core[v] = min_deg
        # Update degrees of neighbours
        for u in np.where(adjacency[v])[0]:
            if not removed[u]:
                current_deg[u] -= 1

    return core.astype(float)


def _compute_effective_degree(adjacency):
    """
    Effective degree: degree(v) - edges_in_N(v)
    where edges_in_N(v) is the number of edges in the induced subgraph on N(v).
    Lower effective degree means the neighbors of v are less interconnected,
    so picking v blocks fewer potential independent set members.
    """
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    
    # M[i,j] = number of common neighbors of i and j
    M = A_int @ A_int.T
    
    # S[v] = sum_u A[v,u] * M[v,u]
    S = (A_int * M).sum(axis=1).astype(float)
    
    # edges_in_N(v) = 0.5 * S[v]
    edges_in_N = 0.5 * S
    
    # effective_degree(v) = degree(v) - edges_in_N(v)
    # Note: edges_in_N counts each edge twice in the sum over neighbors,
    # but we already halved it. Actually let's verify:
    # For vertex v with neighbors {u1, u2, ..., ud}:
    # edges_in_N(v) = number of pairs (ui, uj) with ui~uj
    # = 0.5 * sum_{ui in N(v)} deg_{N(v)}(ui)
    # deg_{N(v)}(ui) = number of neighbors of ui that are also in N(v)
    # = |N(ui) ∩ N(v)| = M[ui, v] (since M[i,j] = |N(i) ∩ N(j)|)
    # So sum_{ui in N(v)} M[ui, v] = sum_u A[v,u] * M[u,v] = sum_u A[v,u] * M[v,u] (symmetric)
    # = S[v]
    # edges_in_N(v) = 0.5 * S[v]
    # This is correct.
    
    return degree - edges_in_N


def _compute_neighborhood_sparsity(adjacency):
    """
    Neighborhood sparsity ratio: fraction of neighbor pairs that are NON-adjacent.
    Higher means neighbors can coexist more in an independent set.
    """
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    
    d_v = degree
    total_pairs = d_v * (d_v - 1.0) / 2.0
    
    # Avoid division by zero
    sparsity_ratio = np.where(total_pairs > 0, 
                              1.0 - edges_in_N / np.maximum(total_pairs, 1.0), 
                              1.0)
    
    return sparsity_ratio


def _compute_mean_neighbor_degree(adjacency):
    """Mean degree of neighbors for each vertex."""
    degree = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degree
    return neighbor_deg_sum / np.maximum(degree, 1.0)


def _search_best_scores(adjacency, time_limit=100.0):
    """
    Search for a good static scoring function by trying various linear combinations
    of degree, neighbor degree sum, coreness, effective degree, and neighborhood sparsity.
    Returns the best scores array found.
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + time_limit

    # Precompute base features
    degree = _compute_degree(adjacency)
    nds = _compute_neighbor_degree_sum(adjacency)
    coreness = _compute_coreness(adjacency)
    effective_deg = _compute_effective_degree(adjacency)
    sparsity_ratio = _compute_neighborhood_sparsity(adjacency)
    mean_neighbor_deg = _compute_mean_neighbor_degree(adjacency)

    # Normalize features to [0, 1] range for stable combination
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    deg_norm = normalize(degree)
    nds_norm = normalize(nds)
    core_norm = normalize(coreness)
    eff_deg_norm = normalize(effective_deg)
    sparsity_norm = normalize(sparsity_ratio)
    mnd_norm = normalize(mean_neighbor_deg)

    best_scores = -degree  # Start with the seed strategy (lowest degree first)
    best_count = _greedy_decode(adjacency, best_scores)

    # Define feature vectors where positive weight means "prefer high value"
    # We want to prefer: low degree, low effective degree, high sparsity, high core, high mean neighbor degree
    # So we use: -deg_norm, -eff_deg_norm, sparsity_norm, core_norm, mnd_norm
    
    features = {
        '-deg': -deg_norm,
        '-eff': -eff_deg_norm,
        'sparsity': sparsity_norm,
        'core': core_norm,
        'mnd': mnd_norm,
    }
    
    feature_names = list(features.keys())
    feature_arrays = [features[name] for name in feature_names]

    # Try structured combinations
    structured_combos = [
        # (w_deg, w_eff, w_sparsity, w_core, w_mnd)
        (-1, 0, 0, 0, 0),       # lowest degree
        (0, -1, 0, 0, 0),       # lowest effective degree
        (0, 0, 1, 0, 0),        # highest sparsity
        (0, 0, 0, 1, 0),        # highest core
        (0, 0, 0, 0, 1),        # highest mean neighbor degree
        (-0.5, -0.5, 0, 0, 0),
        (-0.3, -0.3, 0.2, 0, 0),
        (-0.2, -0.4, 0.2, 0, 0),
        (-0.4, -0.2, 0.2, 0, 0),
        (-0.5, -0.5, 0, 0.2, 0),
        (-0.5, -0.5, 0, 0, 0.2),
        (-0.3, -0.3, 0.2, 0.1, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1),
        (-0.5, -0.5, 0.2, 0.1, 0.1),
        (-0.3, -0.3, 0.3, 0.1, 0.1),
        (-0.2, -0.4, 0.3, 0.1, 0.1),
        (-0.4, -0.2, 0.3, 0.1, 0.1),
        (-0.5, -0.5, 0.3, 0.1, 0.1),
        (-0.3, -0.3, 0.2, 0.2, 0.1),
        (-0.2, -0.4, 0.2, 0.2, 0.1),
        (-0.4, -0.2, 0.2, 0.2, 0.1),
        (-0.5, -0.5, 0.2, 0.2, 0.1),
        (-0.3, -0.3, 0.2, 0.1, 0.2),
        (-0.2, -0.4, 0.2, 0.1, 0.2),
        (-0.4, -0.2, 0.2, 0.1, 0.2),
        (-0.5, -0.5, 0.2, 0.1, 0.2),
    ]

    for weights in structured_combos:
        if time.time() > deadline - 5:
            break
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Now try random perturbations around the best found so far
    # Also try random linear combinations
    rng = np.random.default_rng(42)
    
    num_random_tries = 3000
    for i in range(num_random_tries):
        if time.time() > deadline - 5:
            break
        # Random weights
        w1 = rng.uniform(-1, 0)  # degree: negative means prefer low degree
        w2 = rng.uniform(-1, 0)  # effective degree: negative means prefer low
        w3 = rng.uniform(0, 0.5) # sparsity: positive means prefer high
        w4 = rng.uniform(0, 0.5) # core: positive means prefer high
        w5 = rng.uniform(0, 0.3) # mnd: positive means prefer high
        weights = [w1, w2, w3, w4, w5]
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Local search: perturb the best scores with small random noise and re-evaluate
    # This helps escape local optima in the discrete greedy decoding space
    local_search_steps = 1500
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
