"""Clique partition of two fixed dense random graphs: triangle decomposition + single edges."""
import time

_DEADLINE = 140.0  # wall-clock safety margin below the 150s kill
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _random_graph(n, p, seed):
    """Reproduce the verifier's edge set for (n, p, seed)."""
    import random
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _validate(n, edges, cliques):
    """Every listed set is a clique (>=2) and every edge lies in exactly one clique."""
    edge_set = set(edges)
    seen = set()
    for c in cliques:
        if len(c) < 2:
            return False
        k = len(c)
        for i in range(k):
            a = c[i]
            for j in range(i + 1, k):
                b = c[j]
                e = (a, b) if a < b else (b, a)
                if e not in edge_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edge_set


def _solve_instance(n, p, seed):
    edges = _random_graph(n, p, seed)
    edge_set = set(edges)
    if not edges:
        return []

    # Adjacency bitsets (int masks)
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Group edges by their smaller endpoint
    by_u = {}
    for (u, v) in edges:
        by_u.setdefault(u, []).append(v)

    # Greedy maximal triangle packing
    used = set()
    triangles = []
    for (u, v) in edges:
        if (u, v) in used:
            continue
        # third vertex must be greater than both and adjacent to both
        cand = adj[u] & adj[v]
        cand &= ~((1 << (v + 1)) - 1)  # only w > v
        w = (cand & -cand).bit_length() - 1
        if w < 0:
            continue
        if _elapsed() > _DEADLINE:
            break
        used.add((u, v))
        used.add((u, w) if u < w else (w, u))
        used.add((v, w) if v < w else (w, v))
        triangles.append([u, v, w])

    # Remaining edges become single-edge cliques
    cliques = triangles
    for (u, v) in edges:
        if (u, v) not in used:
            cliques.append([u, v])

    # Safety: verify the partition; fall back to all single-edge cliques if invalid
    if not _validate(n, edges, cliques):
        cliques = [[u, v] for (u, v) in edges]
    return cliques


def construct():
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
