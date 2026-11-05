"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    # adjacency: bool matrix, n x n
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]

    # Degree of each vertex
    deg = A.sum(axis=1).astype(float)

    # Mean degree of neighbors for each vertex.
    # For vertex i, neighbors are A[i, :] (excluding self if present).
    # mean_neighbor_deg[i] = sum(deg[j] for j in N(i)) / max(deg[i],1)
    # Use matrix multiply: (A @ deg) gives sum of neighbor degrees.
    sum_neighbor_deg = A @ deg
    # Avoid division by zero
    safe_deg = np.where(deg > 0, deg, 1.0)
    mean_neighbor_deg = sum_neighbor_deg / safe_deg

    # Core number (degeneracy ordering) computed via a simple peeling.
    # This captures how "central" a vertex is in the densest subgraph.
    # We compute the core number for each vertex.
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    cur_deg = deg.copy().astype(int)

    # Peeling: repeatedly remove the vertex with the smallest current degree
    # among remaining vertices. Its core number is its current degree.
    # For n=200 this is cheap.
    order = np.argsort(cur_deg, kind='stable')
    # We'll do a simplified peeling using a queue-like approach.
    # Track which vertices are still in the graph.
    in_graph = np.ones(n, dtype=bool)
    # Use a bucket approach for efficiency (n is small so even O(n^2) is fine)
    for _ in range(n):
        # Find remaining vertex with min current degree
        candidates = np.where(in_graph)[0]
        if len(candidates) == 0:
            break
        # current degree among remaining
        cand_deg = np.array([cur_deg[c] for c in candidates])
        min_idx = np.argmin(cand_deg)
        v = candidates[min_idx]
        core[v] = cur_deg[v]
        in_graph[v] = False
        # Update degrees of neighbors of v that are still in graph
        nbrs = np.where(A[v] & in_graph)[0]
        for w in nbrs:
            cur_deg[w] -= 1

    # Normalize core to [0,1] for weighting
    max_core = core.max() if core.max() > 0 else 1
    core_norm = core.astype(float) / max_core

    # Score: prefer low degree, low mean-neighbor-degree, low core number.
    # The idea: a vertex is "cheap" to include if it has few neighbors and
    # those neighbors are themselves low-degree (so we don't waste high-value nodes).
    # Combine with core number to break ties toward peripheral (low-core) vertices.
    score = -(deg + 0.5 * mean_neighbor_deg + 0.1 * core_norm)

    return score
# EVOLVE-BLOCK-END
