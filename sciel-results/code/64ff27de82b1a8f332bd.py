"""Guided local search penalty rule for TSP using structural exclusion of toxic edges."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy: Structural Exclusion of Toxic Edges.
    Instead of adding a smooth penalty gradient to all edges based on usage,
    this strategy identifies a small set of "toxic" edges. These are edges that
    are frequently used in past local optima (high edge_n_used) but are also
    geometrically short (low edge_distance). These edges often act as "traps"
    because 2-opt tends to keep them due to their low local cost, even though
    they may prevent the tour from reaching a global optimum by locking in
    a suboptimal local topology.
    
    By setting the cost of these specific edges to a very high value (effectively
    removing them from the feasible set for the next 2-opt run), we force the
    local search to find alternative, potentially longer, connections. This
    structural disruption is more likely to escape the basin of attraction than
    a gentle penalty, which 2-opt might simply overcome by swapping in slightly
    longer non-penalized edges.
    """
    # Check deadline: if we are too close to the limit, return a safe, 
    # minimally perturbed matrix to avoid timeout.
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    D = edge_distance.copy()

    # Ensure we are working with float types for arithmetic
    D = D.astype(np.float64)
    counts = edge_n_used.astype(np.float64)
    dists = edge_distance.astype(np.float64)

    # Identify edges in the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u != v:
            tour_edges.append((u, v))
    
    if not tour_edges:
        return D

    # Calculate a "toxicity" score for each edge in the current tour.
    # Toxicity = Usage Count / (Distance + epsilon)
    # High usage and low distance = High Toxicity.
    # These are the edges the search is "stuck" on.
    toxicities = []
    max_toxicity = 0.0
    for u, v in tour_edges:
        c = counts[u, v]
        d = dists[u, v]
        # Avoid division by zero, though TSP edges shouldn't be 0 in random points
        if d < 1e-9:
            d = 1e-9
        score = c / d
        toxicities.append(score)
        if score > max_toxicity:
            max_toxicity = score

    if max_toxicity == 0:
        return D

    # We want to exclude the most toxic edges.
    # Let's select the top 2-5% of the tour's edges based on toxicity.
    # For N=100, that's 2-5 edges.
    # We sort the tour edges by toxicity and pick the top K.
    # K is chosen to be small to ensure we don't break the tour too much,
    # but enough to force a significant restructure.
    K = max(2, int(0.03 * n))  # ~3 edges for N=100
    
    # Sort indices by toxicity
    sorted_indices = np.argsort(toxicities)[::-1]
    indices_to_exclude = sorted_indices[:K]
    
    # Determine the penalty magnitude.
    # We want the excluded edges to be significantly more expensive than the
    # longest edge in the current tour, so 2-opt will avoid them if possible.
    # But not so expensive that it creates numerical issues or forces a completely
    # random tour that 2-opt can't fix.
    # Let's set the cost to the average edge length * (n / 2) or something similar.
    # Actually, a common heuristic is to set it to a large multiple of the max distance.
    # However, if we set it to infinity, 2-opt might get stuck if it can't form a valid
    # tour without those edges (though 2-opt preserves validity, it just swaps).
    # 2-opt works on the matrix. If an edge is "infinite", 2-opt won't select it for insertion
    # if it can avoid it. But 2-opt evaluates 2-opt moves (swaps). 
    # If we set D[u,v] to a large number, 2-opt will see that using that edge is bad.
    
    # Let's use a penalty that is 1.5 * max_edge_length_in_tour + 1.0 * avg_edge_length
    # This ensures it's clearly worse than any single edge swap.
    tour_len = 0.0
    for u, v in tour_edges:
        tour_len += dists[u, v]
    avg_edge_len = tour_len / n
    max_edge_len = np.max([dists[u, v] for u, v in tour_edges])
    
    # Penalty value: make these edges very unattractive
    # If we set them to a very high value, 2-opt will try to remove them from the tour.
    # Since 2-opt starts from the current tour, and the current tour contains these edges,
    # 2-opt will immediately try to swap them out.
    penalty_value = 10.0 * max_edge_len
    
    for idx in indices_to_exclude:
        u, v = tour_edges[idx]
        D[u, v] = penalty_value
        D[v, u] = penalty_value

    # Ensure finiteness
    if not np.all(np.isfinite(D)):
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
