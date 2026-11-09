"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(scores, adj, n):
    """Fixed greedy decoder: pick vertices in descending score, skip neighbours of picked."""
    order = np.argsort(-scores, kind='stable')
    picked = np.zeros(n, dtype=bool)
    chosen = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if picked[v]:
            continue
        chosen[v] = True
        count += 1
        # mark neighbours as picked (blocked)
        if adj[v].any():
            picked[adj[v]] = True
    return count


def priority(adjacency):
    """
    Assign a score to each vertex. The decoder picks in descending score,
    skipping neighbours of already-picked vertices. We want to maximise the
    number of picked vertices (independent set size).

    Strategy: combine multiple heuristics that are known to work well for
    Maximum Independent Set via greedy decoding:
      1. Low degree (fewer conflicts)
      2. Low "weighted" degree where neighbours with high degree are penalised more
         (a high-degree neighbour will block many other vertices, so being adjacent
         to it is costly)
      3. A small random tie-breaker to avoid degenerate equal-score plateaus

    We use a smooth surrogate: score = -(degree + alpha * sum_{u in N(v)} degree(u))
    This is a 2-hop neighbourhood penalty that captures the "expansion" cost.
    """
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s limit

    adj = np.asarray(adjacency, dtype=np.int32)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    deg = adj.sum(axis=1).astype(np.float64)

    # 2-hop weighted degree: for each v, sum of degrees of its neighbours
    # This is adj @ deg
    two_hop = adj.dot(deg)

    # Primary score: negative (degree + alpha * two_hop)
    # Lower degree and lower neighbour-degree-sum => higher priority
    alpha = 0.5
    base_score = -(deg + alpha * two_hop)

    # Add a tiny deterministic perturbation based on vertex index to break ties
    # without randomness (reproducibility), but also add a small random component
    # to help escape plateaus. Use a fixed seed for reproducibility.
    rng = np.random.RandomState(42)
    jitter = rng.uniform(0, 1e-6, size=n)

    scores = base_score + jitter

    # Check time; if we have budget, try a few alpha values and pick the best
    # by running the greedy decoder on each and selecting the alpha that gives
    # the largest independent set.
    best_scores = scores
    best_count = _greedy_decode(scores, adj, n)

    if time.time() < deadline - 5.0:
        # Try a small grid of alpha values
        for a in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            if time.time() > deadline - 5.0:
                break
            s = -(deg + a * two_hop) + jitter
            c = _greedy_decode(s, adj, n)
            if c > best_count:
                best_count = c
                best_scores = s

    # Also try a pure degree-based score (alpha=0 is already covered above)
    # and a pure two-hop score
    if time.time() < deadline - 5.0:
        s = -two_hop + jitter
        c = _greedy_decode(s, adj, n)
        if c > best_count:
            best_count = c
            best_scores = s

    return best_scores
# EVOLVE-BLOCK-END
