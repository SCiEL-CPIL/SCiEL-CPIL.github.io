"""Rewritten program using a structural change: conflict-radius-based scoring."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def priority(adjacency):
    """
    Assign a score to each vertex such that a fixed greedy decoder (descending score,
    skip neighbours of picked vertices) yields a large independent set.

    Strategy:
      - Compute 'conflict radius' for each vertex: number of edges among its neighbours.
        This measures local clustering density around the vertex.
      - Vertices with low conflict radius, when selected, disable fewer potential future
        candidates than high-density hub vertices.
      - Combine with degree: prefer low-degree vertices with low conflict radius.
      - Use a composite score that balances these two structural features.
    """
    deadline = time.time() + 110  # safety margin before 120s limit

    n = adjacency.shape[0]
    A = adjacency.astype(np.int32)

    # Degree of each vertex
    degree = A.sum(axis=1)

    # Conflict radius: for each vertex v, count edges among its neighbours.
    # For vertex v with neighbours N(v), conflict_radius[v] = number of edges (i,j)
    # where i, j in N(v) and i < j (to avoid double counting, but for scoring we can
    # just count all ordered pairs or use a normalized version).
    #
    # Efficient computation:
    # For each vertex v, conflict_radius[v] = sum_{u in N(v)} (degree[u] - 1) / 2
    # but that's not quite right. Let's think again.
    #
    # Actually, the number of edges among neighbours of v is:
    #   sum_{u in N(v)} sum_{w in N(v), w > u} A[u,w]
    # = (1/2) * sum_{u in N(v)} (sum_{w in N(v)} A[u,w] - A[u,u])
    # = (1/2) * sum_{u in N(v)} (sum_{w in N(v)} A[u,w])
    # since A[u,u] = 0.
    #
    # Note that sum_{w in N(v)} A[u,w] is the number of neighbours of u that are also
    # neighbours of v, i.e., the common neighbours of u and v.
    #
    # So conflict_radius[v] = (1/2) * sum_{u in N(v)} common_neighbours(u, v)
    #
    # This can be computed as: for each edge (u,v), common_neighbours(u,v) contributes
    # to both conflict_radius[u] and conflict_radius[v].
    #
    # Alternatively, we can compute it via matrix multiplication:
    # B = A @ A, where B[u,v] = number of common neighbours of u and v.
    # Then conflict_radius[v] = (1/2) * sum_{u in N(v)} B[u,v] = (1/2) * (B * A).sum(axis=1)
    #
    # Let's compute B = A @ A (this is O(n^3) in worst case but n=200, so fine).
    B = A @ A  # B[u,v] = number of common neighbours of u and v

    # For each vertex v, sum B[u,v] over u in N(v):
    # This is (B * A).sum(axis=1) where * is element-wise.
    conflict_raw = (B * A).sum(axis=1)
    conflict_radius = conflict_raw / 2.0

    # Now we want to prioritize vertices with LOW conflict_radius and LOW degree.
    # Score should be such that higher score = better to pick first.
    # So score = - (alpha * degree + beta * conflict_radius)
    # We need to normalize to make them comparable.

    max_degree = degree.max() if n > 0 else 1
    max_conflict = conflict_radius.max() if n > 0 else 1

    # Normalize to [0, 1]
    if max_degree > 0:
        norm_degree = degree / max_degree
    else:
        norm_degree = degree.astype(float)

    if max_conflict > 0:
        norm_conflict = conflict_radius / max_conflict
    else:
        norm_conflict = conflict_radius.astype(float)

    # Composite score: we want low degree and low conflict radius.
    # Try a weighted sum with emphasis on conflict radius as the structural change.
    alpha = 0.4  # weight for degree
    beta = 0.6   # weight for conflict radius

    score = -(alpha * norm_degree + beta * norm_conflict)

    return score
# EVOLVE-BLOCK-END
