"""Guided local search penalty rule for the TSP - combined multiplicative adaptive penalty."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.

    Strategy:
    - Uses a multiplicative penalty model to preserve the relative geometry of the distance matrix.
    - Strongly penalizes edges in the current local-opt tour, with the penalty magnitude
      scaling with both the edge's usage count (from previous rounds) and the current round number.
    - This forces the subsequent 2-opt to break the most 'stuck' edges first, effectively
      acting as a smooth surrogate for edge exclusion.
    - Applies a smaller global penalty to other frequently-used edges to discourage
      oscillation between similar local optima.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    # Work in float64
    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Extract tour edges and compute scale ---
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))

    # Infer round number from max usage in the entire matrix
    # In the first round, U is all zeros. In round 1, U has 1s for the first tour.
    # In round k, max(U) is roughly k (if the same edges are reused) or less.
    max_used_global = float(U.max()) if U.size > 0 else 0.0
    # If max_used_global is 0, it's the first round (or no usage yet).
    # The evaluator adds current tour edges to U *before* calling this function?
    # "add the current tour's edges to edge_n_used, call your function"
    # So in Round 1, U contains the first tour's edges (count=1). max_used_global = 1.
    # In Round 2, U contains first tour + second tour. max_used_global >= 1.
    # Let's estimate round number.
    # If the tour is completely new, max_used_global might still be 1 from round 1.
    # However, the task says "edge_n_used is a symmetric count matrix accumulated over the rounds".
    # The prompt says: "add the current tour's edges to edge_n_used, call your function".
    # So U includes the current tour.
    
    # A robust way to estimate round: 
    # If max_used_global is 0, return D (shouldn't happen per description, but safe).
    # Let's use max_used_global as a proxy for pressure intensity.
    # Round 1: max U is 1.
    # Round 20: max U could be up to 20 if the same edge is always used.
    
    round_proxy = max(1.0, max_used_global)

    # --- Build multiplicative ratio matrix ---
    # Initialize ratio to 1.0 (no change)
    ratio = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(ratio, 1.0)

    # Identify the max usage specifically within the current tour edges for normalization
    tour_usages = np.array([U[u, v] for u, v in tour_edges], dtype=np.float64)
    max_tour_usage = float(tour_usages.max()) if len(tour_usages) > 0 else 1.0
    if max_tour_usage < 1e-9:
        max_tour_usage = 1.0

    # --- Penalty Parameters ---
    # Base penalty coefficient for tour edges.
    # We want to strongly discourage the current tour edges.
    # In Inspiration 1, 'a' was 0.15 to 0.6.
    # Let's scale 'a' with the round proxy to increase pressure over time.
    # a = 0.5 + 0.05 * log(round_proxy) ? 
    # Let's try a base of 0.8 for round 1, growing slowly.
    # If we set ratio to 1 + 0.8, the distance doubles. 2-opt will likely avoid it.
    # If we set ratio to 1 + 2.0, it triples.
    
    # Adaptive coefficient:
    # Starts strong to break the initial local optimum, and grows to break persistent ones.
    base_alpha = 0.8 * (1.0 + 0.1 * np.log1p(round_proxy))
    
    # Power parameter for usage scaling.
    # p=1.0 is linear. p<1 boosts low usage edges.
    # Since we want to break the *current* tour, and all edges in it have been used at least once in this round,
    # the differentiation comes from *previous* usage.
    # If an edge was in the tour 5 times, it should be penalized more than one used 1 time.
    # Let's use p=1.0 for linear scaling with usage, normalized by max_tour_usage.
    p = 1.0

    # Apply penalty to tour edges
    for (u, v), usage in zip(tour_edges, tour_usages):
        # Normalize usage: usage / max_tour_usage is in [0, 1]
        # If max_tour_usage is 1 (first round), all tour edges have usage 1, so frac=1.
        # If max_tour_usage is 5, the most used edge has frac=1, others < 1.
        frac = usage / max_tour_usage
        # Penalty factor: 1 + alpha * (frac ^ p)
        # This ensures the most used edge gets the highest penalty.
        r = 1.0 + base_alpha * (frac ** p)
        ratio[u, v] = r
        ratio[v, u] = r

    # --- Global Penalty for Non-Tour Used Edges ---
    # Discourage re-forming past local optima by penalizing other high-usage edges.
    # This should be much smaller than the tour penalty to avoid over-constraining.
    global_alpha = 0.1 * (1.0 + 0.05 * np.log1p(round_proxy))
    
    # Soft usage metric for global penalty
    # Use U / (1 + U) to bound the penalty, or just linear normalized?
    # Let's use a soft saturation to keep it stable.
    soft_global_usage = U / (1.0 + U)
    
    # Mask: True where NOT a current-tour edge and NOT diagonal
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)
    
    # Apply global penalty only where mask is True and usage > 0
    # We only apply if the edge has been used before.
    where_global = mask & (U > 0)
    
    # Calculate global ratio
    r_global = 1.0 + global_alpha * soft_global_usage
    ratio = np.where(where_global, r_global, ratio)

    # --- Apply Multiplicative Penalty ---
    result = D * ratio

    # Ensure symmetry
    result = (result + result.T) * 0.5
    
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
