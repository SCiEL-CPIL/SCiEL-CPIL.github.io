"""Rewritten program: complement-density-based priority for MIS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy MIS selection.
    Uses complement-graph density to identify vertices that form
    dense cliques in the complement (isolated clusters in G).
    """
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    n = adjacency.shape[0]
    if n == 0:
        return np.array([])

    # Work with int for speed
    adj = adjacency.astype(np.int8)
    deg = adj.sum(axis=1).astype(np.float64)

    # Complement degree: number of non-neighbors (excluding self)
    comp_deg = (n - 1) - deg

    # --- Multi-scale scoring ---
    # 1) Base: complement degree (vertices with few G-neighbors are good)
    # 2) Local complement density: for each vertex, how dense is its
    #    neighborhood in the complement? High = clique-like in G̅
    # 3) Global: weighted by how many of its neighbors have high degree
    #    (picking a vertex "frees" its neighbors; prefer picking low-degree
    #    neighbors' neighbors less)

    # Complement adjacency (for n=200, this is 200x200, fine)
    comp_adj = 1 - adj - np.eye(n, dtype=np.int8)

    # Local complement density: for vertex i, mean comp-degree of its
    # complement-neighbors. This measures clique density in G̅ around i.
    comp_deg_float = comp_deg.astype(np.float64)
    # Sum of comp-degrees over complement-neighbors
    comp_neighbor_comp_deg_sum = comp_adj.astype(np.float64) @ comp_deg_float
    comp_neighbor_count = comp_deg_float
    # Avoid division by zero
    safe_count = np.where(comp_neighbor_count > 0, comp_neighbor_count, 1.0)
    local_comp_density = comp_neighbor_comp_deg_sum / safe_count

    # Global structural score: prefer vertices whose G-neighbors are low-degree
    # (so we "waste" less by excluding them)
    neighbor_deg_sum = adj.astype(np.float64) @ deg
    safe_deg = np.where(deg > 0, deg, 1.0)
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    # Lower avg_neighbor_deg is better -> invert
    global_score = -avg_neighbor_deg

    # Normalize each component to [0,1] roughly
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    s1 = normalize(comp_deg_float)       # complement degree
    s2 = normalize(local_comp_density)   # local clique density in G̅
    s3 = normalize(-global_score)        # prefer low avg neighbor degree

    # Combine with weights
    score = 0.45 * s1 + 0.35 * s2 + 0.20 * s3

    # --- Iterative refinement: simulate greedy, detect conflicts, adjust ---
    # We do a few rounds of "what if I picked this vertex, who do I block?"
    # and penalize vertices that block many high-value candidates.

    for round_iter in range(3):
        if time.time() > DEADLINE:
            break

        # Simulate greedy selection with current scores
        order = np.argsort(-score)
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)

        for v in order:
            if selected[v] or blocked[v]:
                continue
            selected[v] = True
            # Block neighbors
            nbrs = np.where(adj[v, :])[0]
            blocked[nbrs] = True

        # For each vertex, compute how many "valuable" vertices it blocks
        # Valuable = high score. If picking v blocks many high-score vertices,
        # penalize v.
        # blocked_by[v] = sum of scores of vertices that v would block
        # i.e., for each v, sum of score[u] for u in neighbors(v)
        blocked_value = adj.astype(np.float64) @ score
        # Also account for: if v itself is low-score, it's cheaper to block
        # Actually we want: penalty = sum of (score[u] for u in N(v))
        # Normalize
        bv_norm = normalize(blocked_value)

        # Adjust: subtract a fraction of blocked value
        score = score - 0.15 * bv_norm

        # Re-normalize to keep scale stable
        score = normalize(score)

    # Final: add small tie-breaker based on original degree to ensure
    # deterministic ordering
    score = score - 1e-6 * normalize(deg)

    if time.time() > DEADLINE:
        return score

    return score
# EVOLVE-BLOCK-END
